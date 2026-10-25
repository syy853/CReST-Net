import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clean_state_dict(sd):
    if isinstance(sd, dict):
        for key in ["model", "state_dict", "teacher", "student"]:
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break

    out = {}
    for k, v in sd.items():
        nk = k
        for p in ["module.", "teacher.", "student.", "backbone."]:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _extract_patch_tokens(out):
    if isinstance(out, dict):
        for key in ["x_norm_patchtokens", "patchtokens", "x_prenorm", "tokens"]:
            if key in out:
                x = out[key]
                break
        else:
            raise RuntimeError(f"Unsupported DINO dict keys: {list(out.keys())}")
    else:
        x = out

    if x.dim() != 3:
        raise RuntimeError(f"Expected DINO token tensor B,N,C, got {tuple(x.shape)}")

    n = x.shape[1]
    g = int(math.sqrt(n))
    if g * g == n:
        patch = x
    else:
        patch = x[:, 1:, :]
        n2 = patch.shape[1]
        g = int(math.sqrt(n2))
        if g * g != n2:
            raise RuntimeError(f"DINO patch token count is not square: total={n}, patch={n2}")

    return patch, g


class DINOTeacherObjectnessLoss(nn.Module):
    """
    Training-only frozen DINO objectness teacher.

    It does not change the inference graph. During training, it:
    1) extracts DINO patch features;
    2) builds foreground/background prototypes using GT masks;
    3) generates an objectness soft target from fg-bg similarity;
    4) supervises the saliency prediction with a soft BCE term.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2",
        ckpt_path: str = "pretrained/dinov2_vits14_pretrain.pth",
        input_size: int = 518,
        temperature: float = 0.07,
        min_fg_ratio: float = 0.002,
        min_bg_ratio: float = 0.002,
        teacher_batch_size: int = 1,
    ):
        super().__init__()

        import timm

        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.input_size = int(input_size)
        self.temperature = float(temperature)
        self.min_fg_ratio = float(min_fg_ratio)
        self.min_bg_ratio = float(min_bg_ratio)
        self.teacher_batch_size = int(teacher_batch_size)

        self.teacher = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=self.input_size,
        )

        raw = torch.load(ckpt_path, map_location="cpu")
        sd = _clean_state_dict(raw)
        model_sd = self.teacher.state_dict()

        filtered = {}
        for k, v in sd.items():
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
                filtered[k] = v

        ret = self.teacher.load_state_dict(filtered, strict=False)

        if len(filtered) < 50:
            raise RuntimeError(
                f"Too few DINO keys loaded: {len(filtered)}. "
                f"Check model_name={model_name}, ckpt_path={ckpt_path}"
            )

        self.loaded_keys = len(filtered)
        self.missing_keys = len(ret.missing_keys)
        self.unexpected_keys = len(ret.unexpected_keys)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def build_objectness(self, image: torch.Tensor, gt: torch.Tensor) -> Optional[torch.Tensor]:
        """
        image: B,3,H,W, normalized as the training pipeline currently provides.
        gt:    B,1,H,W or B,H,W, binary/soft GT mask.

        return:
            objectness target B,1,H,W in original image spatial size.
        """
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        gt = gt.float().clamp(0, 1)

        # DINO input. We intentionally do not assume ImageNet mean/std here,
        # because current training tensors may already be normalized by the dataset.
        # This first version tests whether DINO feature affinity itself helps.
        x = F.interpolate(
            image.float(),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        # Run frozen DINO in small chunks to avoid quadratic-attention OOM.
        patches = []
        grids = []
        bs = max(1, int(getattr(self, 'teacher_batch_size', 1)))
        for st in range(0, x.shape[0], bs):
            ed = min(st + bs, x.shape[0])
            with torch.cuda.amp.autocast(enabled=x.is_cuda):
                out = self.teacher.forward_features(x[st:ed])
            patch_i, grid_i = _extract_patch_tokens(out)
            patches.append(patch_i.float())
            grids.append(grid_i)

        if len(set(grids)) != 1:
            raise RuntimeError(f"Inconsistent DINO grids in chunks: {grids}")

        patch = torch.cat(patches, dim=0)
        grid = grids[0]

        feat = patch.transpose(1, 2).reshape(image.shape[0], patch.shape[-1], grid, grid)
        feat = F.normalize(feat, dim=1)

        gt_s = F.interpolate(gt, size=(grid, grid), mode="bilinear", align_corners=False)
        fg_w = gt_s.clamp(0, 1)
        bg_w = (1.0 - gt_s).clamp(0, 1)

        b, _, h, w = fg_w.shape
        area = float(h * w)
        fg_area = fg_w.sum(dim=(2, 3), keepdim=True)
        bg_area = bg_w.sum(dim=(2, 3), keepdim=True)

        valid = (fg_area >= self.min_fg_ratio * area) & (bg_area >= self.min_bg_ratio * area)
        valid = valid.float()

        eps = 1e-6
        fg_proto = (feat * fg_w).sum(dim=(2, 3)) / (fg_area.squeeze(-1).squeeze(-1) + eps)
        bg_proto = (feat * bg_w).sum(dim=(2, 3)) / (bg_area.squeeze(-1).squeeze(-1) + eps)

        fg_proto = F.normalize(fg_proto, dim=1)
        bg_proto = F.normalize(bg_proto, dim=1)

        sim_fg = (feat * fg_proto[:, :, None, None]).sum(dim=1, keepdim=True)
        sim_bg = (feat * bg_proto[:, :, None, None]).sum(dim=1, keepdim=True)

        obj = torch.sigmoid((sim_fg - sim_bg) / max(self.temperature, 1e-6))

        # Invalid masks fall back to GT target to avoid NaN/noisy prototype cases.
        obj = valid * obj + (1.0 - valid) * gt_s

        obj = F.interpolate(obj, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return obj.clamp(0, 1)

    def forward(self, pred, image, gt):
        """
        pred: saliency prediction, either logit or probability.
              This function assumes values outside [0,1] are logits.
        """
        with torch.no_grad():
            target = self.build_objectness(image, gt)

        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

        if pred.detach().min() < 0 or pred.detach().max() > 1:
            prob = torch.sigmoid(pred)
        else:
            prob = pred.clamp(1e-6, 1 - 1e-6)

        loss = F.binary_cross_entropy(prob, target)
        return loss
