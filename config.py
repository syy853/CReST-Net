import os
from pathlib import Path
_CREST_ROOT = Path(__file__).resolve().parent


class Config():
    def __init__(self) -> None:
        # Backbone
        self.proj_root = os.environ.get('CREST_PROJ_ROOT', str(_CREST_ROOT))
        self.bb = 'trans-dinov2'
        self.pvt_weights = ['../bb_weights/pvt_v2_b2.pth', ''][0]

        # Frozen DINOv2 encoder and trainable pyramid adapters.
        self.dinov2_repo = os.environ.get('DINOV2_REPO', str(_CREST_ROOT / 'third_party/dinov2'))
        self.dinov2_checkpoint = os.environ.get('DINOV2_CHECKPOINT', str(_CREST_ROOT / 'pretrained/dinov2_vits14_pretrain.pth'))
        self.dinov2_layer_ids = (2, 5, 8, 11)
        self.freeze_dinov2_backbone = True
        # BN
        self.use_bn = self.bb not in ['cnn-vgg16']
        # Augmentation
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]

        # Components
        self.consensus = 'DPCM'
        self.method_name = 'OHRNetV3_E6_DINOv2_DPT1_DPCM_OVDCD_LVSP_CGPF'

        # Consensus-Guided Prompt Feedback (CGPF).
        # A group-invariant prompt is built from OV-DCD-refined
        # x4 and fed back only to ordinary-LVSP-refined x3.
        self.use_cgpf = True
        self.cgpf_prompt_dim = 64
        self.cgpf_gamma_init = 0.001
        self.cgpf_temperature = 0.20
        self.cgpf_uncertainty_floor = 0.50
        self.cgpf_min_core_support = 0.05
        self.cgpf_detach_consensus = True
        self.paper_name = 'DPCM + OV-DCD + LVSP + CGPF'
        self.dec_blk = ['ResBlk'][0]

        # DPT-C ablation.
        # DINOv2, Pyramid Adapter, DPCM, OV-DCD and DPT1 remain active.
        # OCCR routing calibration is disabled while the shared
        # Objectness Head is retained solely for OV-DCD.
        self.use_dpt_decoder = True
        self.dpt_decoder_dim = 128
        self.dpt_norm_groups = 8
        self.GCAM_metric = 'hpri' if self.consensus else ''

        # CPKM-Net V1: Commonality Prototype Knowledge Aggregation.
        # This replaces MCCL's correlation/non-local GCAM internals while keeping
        # the same x4 consensus interface: B,C,H,W -> B,C,H,W.
        self.cpka_reduction = 4
        self.cpka_topk = 32
        # Training
        self.batch_size = 48
        self.loadN = 2

        # B0-G: during training, each contiguous semantic group is
        # processed independently by DPCM and OV-DCD. During inference,
        # the complete test image group is processed jointly.
        self.group_isolated_consensus = True
        self.dec_att = ['', 'ASPP'][0]
        self.auto_pad = ['', 'fixed', 'adaptive'][0]
        self.optimizer = ['Adam', 'AdamW'][1]
        self.lr = 1e-4
        self.adapter_lr = 1e-4
        self.task_lr = 1e-4
        self.weight_decay = 1e-2
        # Legacy whole-backbone freezing is disabled.
        # DINOv2 freezing is controlled explicitly by
        # freeze_dinov2_backbone and optimizer_utils.py.
        self.freeze = False
        self.lr_decay_epochs = [-20]    # Set to negative N to decay the lr in the last N-th epoch.
        self.forward_per_dataset = True
        # Adv
        self.lambda_adv_g = 0.0
        self.lambda_adv_d = 0.0
        # Loss
        losses = ['sal']
        self.loss = losses[:]
        self.cls_mask_operation = ['x', '+', 'c'][0]
        # Loss + Triplet Loss
        self.lambdas_sal_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'ssim': 1 * 0,          # help contours
            'mse': 150 * 0,         # can smooth the saliency map
            'reg': 100 * 0,
            'triplet': 3 * 0,
        }

        self.db_output_decoder = False
        self.refine = False
        self.db_output_refiner = False
        # Triplet Loss
        self.triplet = ['_x5', 'mask'][:1]
        self.triplet_loss_margin = 0.1

        # CPKM-Net V2: RPMM replaces the original triplet/MCM-style metric loss.
        self.lambda_rpmm = 3.0
        # CPKM-Net V5-a: CCRD / LCPD semantic distillation.
        # Training-time only. It aligns visual foreground prototypes with
        # frozen CLIP text prototypes and separates background prototypes.
        self.use_ccrd = False
        self.lambda_ccrd = 0.0
        self.ccrd_start_epoch = 1
        self.ccrd_fg_weight = 1.0
        self.ccrd_bg_weight = 0.25
        self.ccrd_rank_weight = 0.50
        self.ccrd_margin_bg = 0.05
        self.ccrd_margin_rank = 0.05
        # CPKM-Net V5b: SGIL replaces adversarial integrity learning.
        # No discriminator is used. SGIL enforces whole-object, group-level,
        # background-separation, and CLIP-semantic integrity.
        self.use_sgil = False
        # V7-A: frozen DINO objectness teacher, training only.
                # V7-B: No-CLIP hierarchical prototype reasoning.
        self.use_hpri = False

        # OHR-Net v2: Objectness-Verified Dual Consensus Decomposition.
        self.use_dpcm = True
        self.dpcm_num_prototypes = 4

        # Attention gain used after candidate aggregation:
        # sigmoid(dpcm_temperature * routed_similarity).
        self.dpcm_temperature = 4.0
        self.dpcm_gamma_init = 0.1

        # E1: differentiable routing over the K DPCM candidates.
        # This is deliberately separate from dpcm_temperature.
        self.dpcm_routing_mode = 'soft'
        self.dpcm_routing_temperature = 0.10

        # E3: Unified Objectness-Guided Consensus Refinement.
        #
        # The shared Objectness predictor and OCCR routing calibration
        # are controlled independently in this ablation:
        # 1) the predictor remains active for OV-DCD;
        # 2) candidate-level OCCR calibration is disabled.
        self.use_shared_objectness = True
        self.use_occr = False
        self.use_unified_occr = True
        self.occr_alpha_raw_init = -2.252168461
        self.occr_alpha_max = 2.0
        self.occr_eps = 1e-6

        # Integrate objectness guidance directly into the x4->x3
        # decoder fusion rather than adding a separate decoder module.
        self.use_x3_objectness_guidance = False
        self.x3_guidance_init = 0.10

        # E6: frozen external EVA-02 candidate semantic calibration.
        self.use_eva_semantic_calibration = False
        self.eva_model_name = 'eva02_small_patch14_224'
        self.eva_checkpoint = os.environ.get('EVA_CHECKPOINT', str(_CREST_ROOT / 'pretrained/eva02_small_patch14_224_mim_in22k_state_dict.pth'))
        self.eva_input_size = 224
        self.eva_semantic_raw_delta_init = 0.0
        self.eva_semantic_delta_max = 1.0
        self.eva_semantic_eps = 1e-6

        self.use_ovdcd = True
        self.ovdcd_alpha = 0.7
        self.ovdcd_temperature = 4.0
        self.ovdcd_gamma_init = 0.0

        # Late-Variation Structure Preservation.
        # Only x3 is refined in OV-DCD-uncertain regions.
        self.use_lvsp = True
        self.lvsp_gamma_init = 1e-3
        self.lvsp_eps = 1e-6

        # GT-guided objectness verifier.
        # This supervises the internal objectness head without using an external teacher.
        self.use_gt_objectness = True
        self.lambda_objectness = 0.05
        self.objectness_start_epoch = 1
        self.lambda_hpri = 0.0
        self.hpri_margin_bg = 0.05
        self.hpri_margin_chain = 0.05
        self.hpri_margin_neg = 0.25
        self.hpri_whole_weight = 1.0
        self.hpri_group_weight = 0.5
        self.hpri_bg_weight = 0.3
        self.hpri_chain_weight = 0.2
        self.hpri_inter_weight = 0.2
        self.use_dino_teacher = False
        self.lambda_dino_teacher = 0.0
        self.dino_teacher_model = 'vit_small_patch14_dinov2'
        self.dino_teacher_ckpt = os.environ.get('DINOV2_TEACHER_CHECKPOINT', str(_CREST_ROOT / 'pretrained/dinov2_vits14_pretrain.pth'))
        self.dino_teacher_input_size = 518
        self.dino_teacher_temperature = 0.07
        self.dino_teacher_start_epoch = 1

        # V7-B strict No-CLIP: prevent accidental CLIP text-bank loading.
        self.ccrd_text_banks = {}  # OHR-Net: no CLIP text bank  # OHR-Net strict No-CLIP  # V7-B strict No-CLIP  # V7-B strict No-CLIP  # V7-B strict No-CLIP override

        self.lambda_sgil = 0.0
        self.sgil_start_epoch = 1
        self.sgil_whole_weight = 1.0
        self.sgil_group_weight = 0.5
        self.sgil_bg_weight = 0.3
        self.sgil_text_weight = 0.5
        self.sgil_margin_bg = 0.05

        self.ccrd_text_banks = {}  # OHR-Net: no CLIP text bank  # OHR-Net strict No-CLIP  # V7-B strict No-CLIP  # V7-B strict No-CLIP
        self.rpmm_margin_bg = 0.20
        self.rpmm_margin_neg = 0.25
        self.rpmm_chain_weight = 0.20


        # Intermediate Layers
        self.lambdas_sal_others = {
            'bce': 0,
            'iou': 0.,
            'ssim': 0,
            'mse': 0,
            'reg': 0,
            'triplet': 0,
        }
        self.output_number = 1
        self.loss_sal_layers = 4              # used to be last 4 layers
        self.loss_cls_mask_last_layers = 1         # used to be last 4 layers
        if 'keep in range':
            self.loss_sal_layers = min(self.output_number, self.loss_sal_layers)
            self.loss_cls_mask_last_layers = min(self.output_number, self.loss_cls_mask_last_layers)
            self.output_number = min(self.output_number, max(self.loss_sal_layers, self.loss_cls_mask_last_layers))
            if self.output_number == 1:
                for cri in self.lambdas_sal_others:
                    self.lambdas_sal_others[cri] = 0
        self.conv_after_itp = False
        self.complex_lateral_connection = False

        # to control the quantitive level of each single loss by number of output branches.
        self.loss_cls_mask_ratio_by_last_layers = 4 / self.loss_cls_mask_last_layers
        for loss_sal in self.lambdas_sal_last.keys():
            loss_sal_ratio_by_last_layers = 4 / (int(bool(self.lambdas_sal_others[loss_sal])) * (self.loss_sal_layers - 1) + 1)
            self.lambdas_sal_last[loss_sal] *= loss_sal_ratio_by_last_layers
            self.lambdas_sal_others[loss_sal] *= loss_sal_ratio_by_last_layers
        self.lambda_cls_mask = 2.5 * self.loss_cls_mask_ratio_by_last_layers
        self.lambda_cls = 3.
        self.lambda_contrast = 250.

        # others
        self.self_supervision = False
        self.label_smoothing = False

        self.validation = False
        self.rand_seed = 7
        self.val_last = int(os.environ.get('CREST_VAL_LAST', '50'))
        self.save_step = int(os.environ.get('CREST_SAVE_STEP', '10'))
