import torch
from torch import nn
import torch.nn.functional as F
from hpri_loss import HPRILoss
from torch.autograd import Variable
from math import exp
from config import Config


def pairwise_distance_torch(embeddings, device=torch.device("cuda")):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def TripletSemiHardLoss(y_pred, y_true, device=torch.device("cuda"), margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    labels = torch.reshape(labels, [labels.shape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, ipt, target, **kwargs):
        return TripletSemiHardLoss(ipt, target, margin=self.margin)


class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        # return IoU/b
        return IoU


class ThrReg_loss(torch.nn.Module):
    def __init__(self):
        super(ThrReg_loss, self).__init__()

    def forward(self, pred, gt=None):
        return torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))



class RPMMLoss(nn.Module):
    """
    CPKM-Net V2: Reasoning-guided Prototype Memory Mixer.

    This replaces the original triplet/MCM-style metric branch.
    It builds a visual reasoning chain:
        foreground prototype -> ambiguous prototype -> background prototype

    A lightweight selective state update is used to mix prototype states.
    This is Mamba-style selective state reasoning implemented in pure PyTorch;
    it does not require mamba_ssm or causal_conv1d.
    """
    def __init__(self, margin_bg=0.20, margin_neg=0.25, chain_weight=0.20, eps=1e-6):
        super(RPMMLoss, self).__init__()
        self.margin_bg = margin_bg
        self.margin_neg = margin_neg
        self.chain_weight = chain_weight
        self.eps = eps

    def _weighted_proto(self, feat, weight):
        # feat: B,C,h,w; weight: B,1,h,w
        weight = weight.clamp(min=0.0)
        proto = (feat * weight).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + self.eps)
        return F.normalize(proto, p=2, dim=1)

    def _selective_state_mix(self, seq):
        # seq: B,T,C. Deterministic bidirectional selective state mixer.
        B, T, C = seq.shape

        state = torch.zeros(B, C, device=seq.device, dtype=seq.dtype)
        outs_f = []
        for t in range(T):
            cur = seq[:, t]
            gate = torch.sigmoid((cur * state).sum(dim=1, keepdim=True))
            state = gate * cur + (1.0 - gate) * state
            outs_f.append(state)

        state = torch.zeros(B, C, device=seq.device, dtype=seq.dtype)
        outs_b = []
        for t in reversed(range(T)):
            cur = seq[:, t]
            gate = torch.sigmoid((cur * state).sum(dim=1, keepdim=True))
            state = gate * cur + (1.0 - gate) * state
            outs_b.append(state)
        outs_b = list(reversed(outs_b))

        mixed = torch.stack([(a + b) * 0.5 for a, b in zip(outs_f, outs_b)], dim=1)
        return F.normalize(mixed, p=2, dim=2)

    def _pairwise_group_loss(self, fg, labels):
        # fg: B,C normalized.
        B = fg.shape[0]
        if labels is None or labels.numel() != B:
            # fallback: split the batch into two pseudo-groups
            half = max(B // 2, 1)
            labels = torch.zeros(B, device=fg.device, dtype=torch.long)
            labels[half:] = 1
        else:
            labels = labels.to(fg.device).view(-1)

        unique = labels.unique()
        intra_loss = fg.new_tensor(0.0)
        intra_count = 0

        group_protos = []
        for u in unique:
            idx = torch.where(labels == u)[0]
            if idx.numel() == 0:
                continue
            gp = F.normalize(fg[idx].mean(dim=0, keepdim=True), p=2, dim=1)
            group_protos.append(gp)

            if idx.numel() > 1:
                cur = fg[idx]
                sim = cur @ cur.t()
                mask = ~torch.eye(idx.numel(), device=fg.device, dtype=torch.bool)
                intra_loss = intra_loss + (1.0 - sim[mask]).mean()
                intra_count += 1

        if intra_count > 0:
            intra_loss = intra_loss / intra_count

        inter_loss = fg.new_tensor(0.0)
        inter_count = 0
        if len(group_protos) > 1:
            gps = torch.cat(group_protos, dim=0)
            sim = gps @ gps.t()
            mask = ~torch.eye(gps.shape[0], device=fg.device, dtype=torch.bool)
            inter_loss = F.relu(sim[mask] - self.margin_neg).mean()
            inter_count += 1

        return intra_loss, inter_loss

    def forward(self, feat, sal_logit, labels=None):
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        if feat is None or feat.dim() != 4:
            return sal_logit.new_tensor(0.0)

        B, C, h, w = feat.shape

        # Use saliency prediction as soft evidence. Detach to keep this branch
        # focused on representation shaping rather than destabilizing mask learning.
        sal = torch.sigmoid(sal_logit.detach())
        sal = F.interpolate(sal, size=(h, w), mode='bilinear', align_corners=True)

        fg_w = sal
        bg_w = 1.0 - sal
        amb_w = 1.0 - (2.0 * sal - 1.0).abs()

        fg = self._weighted_proto(feat, fg_w)
        amb = self._weighted_proto(feat, amb_w)
        bg = self._weighted_proto(feat, bg_w)

        # Visual reasoning chain: common foreground -> ambiguous area -> background rejection.
        seq = torch.stack([fg, amb, bg], dim=1)
        mixed = self._selective_state_mix(seq)
        fg_m, amb_m, bg_m = mixed[:, 0], mixed[:, 1], mixed[:, 2]

        # Foreground should be separated from background.
        loss_bg = F.relu((fg_m * bg_m).sum(dim=1) + self.margin_bg).mean()

        # Ambiguous prototype should stay between fg and bg, not collapse to bg.
        sim_fa = (fg_m * amb_m).sum(dim=1)
        sim_fb = (fg_m * bg_m).sum(dim=1)
        loss_chain = F.relu(sim_fb - sim_fa + 0.10).mean()

        # Same group foreground prototypes should be consistent; different groups should separate.
        loss_intra, loss_inter = self._pairwise_group_loss(fg_m, labels)

        loss = loss_intra + loss_bg + 0.5 * loss_inter + self.chain_weight * loss_chain
        return loss

class SalLoss(nn.Module):
    """
    IoU loss for outputs in [1:] scales.
    """
    def __init__(self):
        super(SalLoss, self).__init__()
        self.config = Config()
        self.lambdas_sal_last = self.config.lambdas_sal_last
        self.lambdas_sal_others = self.config.lambdas_sal_others
        self.triplet_loss = ['vanilla', 'semi_hard'][0]
        self.use_rpmm = getattr(self.config, 'GCAM_metric', '') == 'rpmm'
        self.use_hpri = getattr(self.config, 'GCAM_metric', '') == 'hpri'
        if self.use_rpmm:
            self.criterion_rpmm = RPMMLoss(
                margin_bg=getattr(self.config, 'rpmm_margin_bg', 0.20),
                margin_neg=getattr(self.config, 'rpmm_margin_neg', 0.25),
                chain_weight=getattr(self.config, 'rpmm_chain_weight', 0.20)
            )
        if self.use_hpri:
            self.criterion_hpri = HPRILoss(
                margin_bg=getattr(self.config, 'hpri_margin_bg', 0.05),
                margin_chain=getattr(self.config, 'hpri_margin_chain', 0.05),
                margin_neg=getattr(self.config, 'hpri_margin_neg', 0.25),
                whole_weight=getattr(self.config, 'hpri_whole_weight', 1.0),
                group_weight=getattr(self.config, 'hpri_group_weight', 0.5),
                bg_weight=getattr(self.config, 'hpri_bg_weight', 0.3),
                chain_weight=getattr(self.config, 'hpri_chain_weight', 0.2),
                inter_weight=getattr(self.config, 'hpri_inter_weight', 0.2),
            )

        self.criterions_last = {}
        if 'bce' in self.lambdas_sal_last and self.lambdas_sal_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
        if 'iou' in self.lambdas_sal_last and self.lambdas_sal_last['iou']:
            self.criterions_last['iou'] = IoU_loss()
        if 'ssim' in self.lambdas_sal_last and self.lambdas_sal_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mse' in self.lambdas_sal_last and self.lambdas_sal_last['mse']:
            self.criterions_last['mse'] = nn.MSELoss()
        if 'reg' in self.lambdas_sal_last and self.lambdas_sal_last['reg']:
            self.criterions_last['reg'] = ThrReg_loss()
        if 'triplet' in self.lambdas_sal_last and self.lambdas_sal_last['triplet']:
            margin = self.config.triplet_loss_margin
            if self.triplet_loss == 'vanilla':
                self.criterion_triplet = nn.TripletMarginLoss(margin=margin)
            elif self.criterion_triplet == 'semi_hard':
                self.criterion_triplet = TripletLoss(margin=margin)

        self.criterions_others = {}
        if 'bce' in self.lambdas_sal_others and self.lambdas_sal_others['bce']:
            self.criterions_others['bce'] = nn.BCELoss()
        if 'iou' in self.lambdas_sal_others and self.lambdas_sal_others['iou']:
            self.criterions_others['iou'] = IoU_loss()
        if 'ssim' in self.lambdas_sal_others and self.lambdas_sal_others['ssim']:
            self.criterions_others['ssim'] = SSIMLoss()
        if 'mse' in self.lambdas_sal_others and self.lambdas_sal_others['mse']:
            self.criterions_others['mse'] = nn.MSELoss()

    def forward(self, scaled_preds, gt, norm_features=None, labels=None):
        loss = 0
        for idx_output, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.functional.interpolate(pred_lvl, size=gt.shape[2:], mode='bilinear', align_corners=True)
            if idx_output == len(scaled_preds) - 1:
                if not(self.config.db_output_refiner or (not self.config.refine and self.config.db_output_decoder)):
                    pred_lvl = pred_lvl.sigmoid()
                for criterion_name, criterion in self.criterions_last.items():
                    loss += criterion(pred_lvl, gt) * self.lambdas_sal_last[criterion_name]
                # loss_outside = self.criterions_last['iou'](pred_lvl * (1 - gt), gt * (1 - gt)) * self.lambdas_sal_last['iou'] * 2
                # loss_inside = self.criterions_last['bce'](pred_lvl * gt, gt) * self.lambdas_sal_last['bce'] * 2
                # loss_inside += self.criterions_last['mse'](pred_lvl * gt, gt) * self.lambdas_sal_last['mse'] * 2
                # loss += (loss_outside + loss_inside)
            else:
                if not (self.config.refine and self.config.db_output_decoder and idx_output == len(scaled_preds) - 2):
                    pred_lvl = pred_lvl.sigmoid()
                for criterion_name, criterion in self.criterions_others.items():
                    loss += criterion(pred_lvl, gt) * self.lambdas_sal_others[criterion_name]
        if getattr(self, 'use_hpri', False) and norm_features is not None:
            hpri_loss = self.criterion_hpri(norm_features, scaled_preds[-1], gt, labels)
            hpri_loss = hpri_loss * getattr(self.config, 'lambda_hpri', 3.0)
            loss += hpri_loss
            return loss, hpri_loss

        if getattr(self, 'use_rpmm', False) and norm_features is not None:
            rpmm_loss = self.criterion_rpmm(norm_features, scaled_preds[-1], labels)
            rpmm_loss = rpmm_loss * getattr(self.config, 'lambda_rpmm', 3.0)
            loss += rpmm_loss
            return loss, rpmm_loss

        if self.lambdas_sal_last['triplet'] and norm_features is not None:
            triplet_loss = 0
            for norm_feature in norm_features:
                # vanilla triplet loss in PyTorch
                if self.triplet_loss == 'vanilla':
                    num_feature_per_group = norm_feature.shape[0] // 2
                    feat_A = norm_feature[:num_feature_per_group]
                    feat_B = norm_feature[num_feature_per_group:]
                    # A/2 - A/2 - B/2
                    loss_triplet_ancA = self.criterion_triplet(feat_A[:num_feature_per_group//2], feat_A[-(num_feature_per_group//2):], feat_B[:num_feature_per_group//2])
                    loss_triplet_ancB = self.criterion_triplet(feat_B[:num_feature_per_group//2], feat_B[-(num_feature_per_group//2):], feat_A[:num_feature_per_group//2])
                    triplet_loss += (loss_triplet_ancA + loss_triplet_ancB)
                elif self.triplet_loss == 'semi_hard':
                    triplet_loss += self.criterion_triplet(norm_feature, labels)
            triplet_loss = triplet_loss * self.config.lambdas_sal_last['triplet'] / len(norm_features)
            loss += triplet_loss
            return loss, triplet_loss
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x, y):
    ssim = torch.mean(SSIM(x,y))
    return ssim
