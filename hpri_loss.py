import torch
import torch.nn as nn
import torch.nn.functional as F


class HPRILoss(nn.Module):
    """
    H-PRI: Hierarchical Prototype Reasoning and Integrity Learning.

    This is a No-CLIP replacement for the previous RPMM + SGIL combination.
    It uses visual prototypes only:
      1) predicted foreground prototype;
      2) GT whole-object prototype;
      3) group-level object prototype;
      4) ambiguous-region prototype;
      5) GT background prototype.

    It does not load CLIP, text banks, or any language prior.
    """

    def __init__(
        self,
        margin_bg=0.05,
        margin_chain=0.05,
        margin_neg=0.25,
        whole_weight=1.0,
        group_weight=0.5,
        bg_weight=0.3,
        chain_weight=0.2,
        inter_weight=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.margin_bg = float(margin_bg)
        self.margin_chain = float(margin_chain)
        self.margin_neg = float(margin_neg)

        self.whole_weight = float(whole_weight)
        self.group_weight = float(group_weight)
        self.bg_weight = float(bg_weight)
        self.chain_weight = float(chain_weight)
        self.inter_weight = float(inter_weight)

        self.eps = float(eps)

    def _as_prob(self, sal):
        if sal.detach().min() < 0 or sal.detach().max() > 1:
            return torch.sigmoid(sal)
        return sal.clamp(0, 1)

    def _weighted_proto(self, feat, weight):
        if weight.dim() == 3:
            weight = weight.unsqueeze(1)
        weight = weight.float().clamp(0, 1)

        denom = weight.sum(dim=(2, 3)).clamp_min(self.eps)
        proto = (feat * weight).sum(dim=(2, 3)) / denom
        return F.normalize(proto, p=2, dim=1)

    def _group_proto(self, proto, labels):
        B = proto.shape[0]
        if labels is None:
            return proto.mean(dim=0, keepdim=True).expand_as(proto)

        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=proto.device)
        labels = labels.to(device=proto.device).view(-1)

        if labels.numel() != B:
            return proto.mean(dim=0, keepdim=True).expand_as(proto)

        out = torch.zeros_like(proto)
        for y in labels.unique():
            mask = labels == y
            out[mask] = proto[mask].mean(dim=0, keepdim=True)
        return out

    def _inter_group_loss(self, proto, labels):
        B = proto.shape[0]
        if labels is None or B <= 1:
            return proto.new_tensor(0.0)

        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, device=proto.device)
        labels = labels.to(device=proto.device).view(-1)

        if labels.numel() != B:
            return proto.new_tensor(0.0)

        sim = torch.matmul(proto, proto.t())
        eye = torch.eye(B, dtype=torch.bool, device=proto.device)

        same = labels[:, None] == labels[None, :]
        same = same & (~eye)
        diff = labels[:, None] != labels[None, :]

        loss_same = proto.new_tensor(0.0)
        loss_diff = proto.new_tensor(0.0)

        if same.any():
            loss_same = (1.0 - sim[same]).mean()
        if diff.any():
            loss_diff = F.relu(sim[diff] - self.margin_neg).mean()

        return loss_same + loss_diff

    def forward(self, feat, sal_logit, gt, labels=None):
        """
        feat:      B,C,H,W high-level feature, normally x4 / norm_features.
        sal_logit: B,1,h,w prediction.
        gt:        B,1,H0,W0 or B,H0,W0 ground truth.
        labels:    group/category labels for inter-group reasoning.
        """
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        if feat.dim() != 4:
            raise RuntimeError(f"HPRILoss expects feat B,C,H,W, got {tuple(feat.shape)}")

        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if sal_logit.dim() == 3:
            sal_logit = sal_logit.unsqueeze(1)

        B, C, H, W = feat.shape

        gt_small = F.interpolate(gt.float(), size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
        prob = self._as_prob(sal_logit)
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)

        bg_mask = (1.0 - gt_small).clamp(0, 1)

        # Ambiguous region is high where the prediction is uncertain.
        amb_mask = (prob * (1.0 - prob) * 4.0).clamp(0, 1)

        pred_proto = self._weighted_proto(feat, prob)
        gt_proto = self._weighted_proto(feat, gt_small).detach()
        bg_proto = self._weighted_proto(feat, bg_mask).detach()
        amb_proto = self._weighted_proto(feat, amb_mask).detach()

        group_proto = self._group_proto(gt_proto, labels).detach()

        sim_whole = (pred_proto * gt_proto).sum(dim=1)
        sim_group = (pred_proto * group_proto).sum(dim=1)
        sim_bg = (pred_proto * bg_proto).sum(dim=1)
        sim_amb = (pred_proto * amb_proto).sum(dim=1)
        sim_amb_bg = (amb_proto * bg_proto).sum(dim=1)

        # Integrity: predicted foreground should match complete object prototypes.
        loss_whole = (1.0 - sim_whole).mean()
        loss_group = (1.0 - sim_group).mean()

        # Rejection: predicted foreground should be separated from background.
        loss_bg = F.relu(sim_bg - self.margin_bg).mean()

        # Hierarchical chain:
        # foreground should be closer to ambiguous than to background;
        # ambiguous should not collapse into background.
        loss_chain_1 = F.relu(self.margin_chain + sim_bg - sim_amb).mean()
        loss_chain_2 = F.relu(self.margin_chain + sim_amb_bg - sim_amb).mean()
        loss_chain = loss_chain_1 + 0.5 * loss_chain_2

        # Inter-group: prototypes in same group compact, different groups separable.
        loss_inter = self._inter_group_loss(pred_proto, labels)

        total = (
            self.whole_weight * loss_whole
            + self.group_weight * loss_group
            + self.bg_weight * loss_bg
            + self.chain_weight * loss_chain
            + self.inter_weight * loss_inter
        )

        return total
