import os
from collections import OrderedDict
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import resnet50

from models.modules import ResBlk, ObjectnessVerifiedDCD, DiversePrototypeConsensusMining
from models.pvt import pvt_v2_b2
from models.backbones import DINOv2PyramidAdapter
from models.dpt_decoder import DPTDecoder
from models.lvsp import LateVariationStructurePreservation
from models.eva_runtime import FrozenEVARuntime
from models.consensus_prompt_feedback import ConsensusGuidedPromptFeedback
from config import Config


class GCoNet(nn.Module):
    def __init__(self):
        super(GCoNet, self).__init__()
        self.config = Config()
        bb = self.config.bb
        if bb == 'cnn-vgg16':
            bb_net = list(vgg16(pretrained=True).children())[0]
            bb_convs = OrderedDict({
                'conv1': bb_net[:4],
                'conv2': bb_net[4:9],
                'conv3': bb_net[9:16],
                'conv4': bb_net[16:23]
            })
        elif bb == 'cnn-vgg16bn':
            bb_net = list(vgg16_bn(pretrained=True).children())[0]
            bb_convs = OrderedDict({
                'conv1': bb_net[:6],
                'conv2': bb_net[6:13],
                'conv3': bb_net[13:23],
                'conv4': bb_net[23:33]
            })
        elif bb == 'cnn-resnet50':
            bb_net = list(resnet50(pretrained=True).children())
            bb_convs = OrderedDict({
                'conv1': nn.Sequential(*bb_net[0:3]),
                'conv2': bb_net[4],
                'conv3': bb_net[5],
                'conv4': bb_net[6]
            })
        elif bb == 'trans-pvt':
            self.bb = pvt_v2_b2()
            if self.config.pvt_weights:
                if os.path.exists(self.config.pvt_weights):
                    save_model = torch.load(self.config.pvt_weights)
                    model_dict = self.bb.state_dict()
                    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
                    model_dict.update(state_dict)
                    self.bb.load_state_dict(model_dict)
                else:
                    print("Warning: We cannot load the PVT backbone weights.")
                    print("\tIf you are testing/eval, it's okay.")
                    print("\tIf you are training, save it at {}.".format(self.config.pvt_weights))
        elif bb == 'trans-dinov2':
            self.bb = DINOv2PyramidAdapter(
                dinov2_repo=getattr(
                    self.config,
                    'dinov2_repo',
                ),
                checkpoint_path=getattr(
                    self.config,
                    'dinov2_checkpoint',
                ),
                layer_ids=tuple(
                    getattr(
                        self.config,
                        'dinov2_layer_ids',
                        (2, 5, 8, 11),
                    )
                ),
                freeze_backbone=getattr(
                    self.config,
                    'freeze_dinov2_backbone',
                    True,
                ),
            )

        if 'cnn-' in bb:
            self.bb = nn.Sequential(bb_convs)
        lateral_channels_in = {
            'cnn-vgg16': [512, 256, 128, 64],
            'cnn-vgg16bn': [512, 256, 128, 64],
            'cnn-resnet50': [1024, 512, 256, 64],
            'trans-pvt': [512, 320, 128, 64],
            'trans-dinov2': [512, 320, 128, 64],
        }

        # OHR-Net v3: Diverse Prototype Consensus Mining.
        # DPCM replaces the original co-attention layer. It mines multiple
        # candidate object-consensus prototypes from high-level features.
        self.co_x4 = DiversePrototypeConsensusMining(
            channel_in=lateral_channels_in[bb][0],
            num_prototypes=getattr(self.config, 'dpcm_num_prototypes', 4),
            temperature=getattr(self.config, 'dpcm_temperature', 4.0),
            gamma_init=getattr(self.config, 'dpcm_gamma_init', 0.1),
            routing_mode=getattr(
                self.config,
                'dpcm_routing_mode',
                'hard',
            ),
            routing_temperature=getattr(
                self.config,
                'dpcm_routing_temperature',
                0.1,
            ),
            use_shared_objectness=getattr(
                self.config,
                'use_shared_objectness',
                bool(
                    getattr(
                        self.config,
                        'use_occr',
                        False,
                    )
                    or getattr(
                        self.config,
                        'use_ovdcd',
                        False,
                    )
                    or getattr(
                        self.config,
                        'use_x3_objectness_guidance',
                        False,
                    )
                ),
            ),
            use_occr=getattr(
                self.config,
                'use_occr',
                False,
            ),
            occr_alpha_raw_init=getattr(
                self.config,
                'occr_alpha_raw_init',
                -6.0,
            ),
            occr_alpha_max=getattr(
                self.config,
                'occr_alpha_max',
                2.0,
            ),
            occr_eps=getattr(
                self.config,
                'occr_eps',
                1e-6,
            ),
            use_eva_semantic=getattr(
                self.config,
                'use_eva_semantic_calibration',
                False,
            ),
            eva_semantic_raw_delta_init=getattr(
                self.config,
                'eva_semantic_raw_delta_init',
                0.0,
            ),
            eva_semantic_delta_max=getattr(
                self.config,
                'eva_semantic_delta_max',
                1.0,
            ),
            eva_semantic_eps=getattr(
                self.config,
                'eva_semantic_eps',
                1e-6,
            ),
        ) if getattr(self.config, 'use_dpcm', True) else nn.Identity()

        # OHR-Net v3: Objectness-Verified Dual Consensus Decomposition.
        # It decomposes the DPCM candidate consensus into object/background consensus before decoding.
        self.ovdcd = ObjectnessVerifiedDCD(
            lateral_channels_in[bb][0],
            alpha=getattr(self.config, 'ovdcd_alpha', 0.7),
            temperature=getattr(self.config, 'ovdcd_temperature', 4.0),
            gamma_init=getattr(self.config, 'ovdcd_gamma_init', 0.0)
        ) if getattr(self.config, 'use_ovdcd', False) else None

        self.lvsp = (
            LateVariationStructurePreservation(
                x3_channels=lateral_channels_in[bb][1],
                x4_channels=lateral_channels_in[bb][0],
                gamma_init=getattr(
                    self.config,
                    'lvsp_gamma_init',
                    1e-3,
                ),
                eps=getattr(
                    self.config,
                    'lvsp_eps',
                    1e-6,
                ),
            )
            if getattr(
                self.config,
                'use_lvsp',
                False,
            )
            else None
        )

        # Consensus-Guided Prompt Feedback. It is placed
        # after ordinary LVSP and before DPT1 decoding.
        self.cgpf = ConsensusGuidedPromptFeedback(
            x3_channels=320,
            x4_channels=512,
            prompt_dim=getattr(
                self.config,
                'cgpf_prompt_dim',
                64,
            ),
            gamma_init=getattr(
                self.config,
                'cgpf_gamma_init',
                0.001,
            ),
            temperature=getattr(
                self.config,
                'cgpf_temperature',
                0.20,
            ),
            uncertainty_floor=getattr(
                self.config,
                'cgpf_uncertainty_floor',
                0.50,
            ),
            min_core_support=getattr(
                self.config,
                'cgpf_min_core_support',
                0.05,
            ),
            detach_consensus=getattr(
                self.config,
                'cgpf_detach_consensus',
                True,
            ),
        ) if getattr(
            self.config,
            'use_cgpf',
            False,
        ) else None

        self._last_aux = {}

        if getattr(
            self.config,
            'use_eva_semantic_calibration',
            False,
        ):
            eva_runtime = FrozenEVARuntime(
                checkpoint_path=getattr(
                    self.config,
                    'eva_checkpoint',
                ),
                model_name=getattr(
                    self.config,
                    'eva_model_name',
                    'eva02_small_patch14_224',
                ),
                input_size=getattr(
                    self.config,
                    'eva_input_size',
                    224,
                ),
            )
            object.__setattr__(
                self,
                '_eva_runtime',
                eva_runtime,
            )
        else:
            object.__setattr__(
                self,
                '_eva_runtime',
                None,
            )

        # E3: integrate shared-objectness guidance into the existing
        # x4->x3 decoder fusion. This is not a separate decoder block.
        if getattr(
            self.config,
            'use_x3_objectness_guidance',
            False,
        ):
            self.x3_guidance_strength = nn.Parameter(
                torch.tensor(
                    float(
                        getattr(
                            self.config,
                            'x3_guidance_init',
                            0.10,
                        )
                    )
                )
            )
        else:
            self.register_parameter(
                'x3_guidance_strength',
                None,
            )

        self._last_x3_objectness_gate = None

        self.use_dpt_decoder = bool(
            getattr(
                self.config,
                'use_dpt_decoder',
                False,
            )
        )

        if self.use_dpt_decoder:
            # Input order is x1, x2, x3, x4.
            self.dpt_decoder = DPTDecoder(
                in_channels=(
                    lateral_channels_in[bb][3],
                    lateral_channels_in[bb][2],
                    lateral_channels_in[bb][1],
                    lateral_channels_in[bb][0],
                ),
                hidden_dim=int(
                    getattr(
                        self.config,
                        'dpt_decoder_dim',
                        128,
                    )
                ),
                norm_groups=int(
                    getattr(
                        self.config,
                        'dpt_norm_groups',
                        8,
                    )
                ),
            )
        else:
            if self.config.dec_blk == 'ResBlk':
                DecBlk = ResBlk
            else:
                raise ValueError(
                    'Unsupported decoder block: {}'.format(
                        self.config.dec_blk
                    )
                )

            self.top_layer = DecBlk(
                lateral_channels_in[bb][0],
                lateral_channels_in[bb][1],
            )

            self.dec_layer4 = DecBlk(
                lateral_channels_in[bb][1],
                lateral_channels_in[bb][1],
            )
            self.lat_layer4 = nn.Conv2d(
                lateral_channels_in[bb][1],
                lateral_channels_in[bb][1],
                1,
                1,
                0,
            )

            self.dec_layer3 = DecBlk(
                lateral_channels_in[bb][1],
                lateral_channels_in[bb][2],
            )
            self.lat_layer3 = nn.Conv2d(
                lateral_channels_in[bb][2],
                lateral_channels_in[bb][2],
                1,
                1,
                0,
            )

            self.dec_layer2 = DecBlk(
                lateral_channels_in[bb][2],
                lateral_channels_in[bb][3],
            )
            self.lat_layer2 = nn.Conv2d(
                lateral_channels_in[bb][3],
                lateral_channels_in[bb][3],
                1,
                1,
                0,
            )

            self.dec_layer1 = DecBlk(
                lateral_channels_in[bb][3],
                lateral_channels_in[bb][3] // 2,
            )
            self.conv_out1 = nn.Sequential(
                nn.Conv2d(
                    lateral_channels_in[bb][3] // 2,
                    1,
                    1,
                    1,
                    0,
                )
            )


    def get_aux_outputs(self):
        return getattr(self, '_last_aux', {})

    def _run_consensus_once(
        self,
        x4,
        eva_feature_map=None,
    ):
        """
        Run unified shared-objectness consensus refinement on one
        isolated semantic group.

        Stage 1:
            DPCM generates candidate prototypes and the shared
            objectness logit. OCCR calibrates candidate routing.

        Stage 2:
            OV-DCD reuses the same objectness logit for pixel-level
            object/background consensus verification.
        """
        aux = {}
        shared_objectness_logit = None

        if self.config.consensus:
            x4 = self.co_x4(
                x4,
                eva_feature_map=eva_feature_map,
            )

            shared_objectness_logit = getattr(
                self.co_x4,
                'last_shared_objectness_logit',
                None,
            )

            if shared_objectness_logit is None:
                raise RuntimeError(
                    "Unified OCCR did not produce a "
                    "shared objectness logit."
                )

            # Keep the original training interface. train.py already
            # supervises the key 'objectness_logit'.
            aux['objectness_logit'] = (
                shared_objectness_logit
            )

        if self.ovdcd is None:
            return x4, aux

        if shared_objectness_logit is None:
            raise RuntimeError(
                "Shared-objectness OV-DCD cannot run "
                "without the shared objectness logit."
            )

        x4 = self.ovdcd(
            x4,
            shared_objectness_logit,
        )

        aux.update({
            'object_consensus':
                self.ovdcd.last_object_consensus,
            'background_consensus':
                self.ovdcd.last_background_consensus,
            'dual_attention':
                self.ovdcd.last_dual_attention,
        })

        return x4, aux

    def _apply_consensus_pipeline(
        self,
        x4,
        eva_feature_map=None,
    ):
        """
        Apply semantic-group isolation during training.

        Training:
            [group_1 | ... | group_loadN]
            -> process each group independently
            -> concatenate in the original order.

        Evaluation:
            process the complete test image group jointly.
        """
        load_n = int(
            getattr(self.config, 'loadN', 1)
        )

        use_group_isolation = bool(
            getattr(
                self.config,
                'group_isolated_consensus',
                False,
            )
        )

        should_isolate = (
            self.training
            and use_group_isolation
            and load_n > 1
        )

        if not should_isolate:
            self._last_group_sizes = (
                int(x4.shape[0]),
            )
            return self._run_consensus_once(
                x4,
                eva_feature_map=eva_feature_map,
            )

        batch_size = int(x4.shape[0])

        if batch_size % load_n != 0:
            raise RuntimeError(
                "The x4 batch cannot be divided into equal "
                "semantic groups: "
                f"batch_size={batch_size}, loadN={load_n}."
            )

        group_size = batch_size // load_n

        if group_size < 1:
            raise RuntimeError(
                "The semantic group size must be positive: "
                f"batch_size={batch_size}, loadN={load_n}."
            )

        group_chunks = torch.split(
            x4,
            group_size,
            dim=0,
        )

        if eva_feature_map is None:
            eva_group_chunks = [None] * len(group_chunks)
        else:
            if eva_feature_map.shape[0] != batch_size:
                raise RuntimeError(
                    'x4/EVA batch mismatch: {} vs {}'.format(
                        batch_size,
                        eva_feature_map.shape[0],
                    )
                )

            eva_group_chunks = torch.split(
                eva_feature_map,
                group_size,
                dim=0,
            )

        if len(group_chunks) != load_n:
            raise RuntimeError(
                "Unexpected semantic-group count after split: "
                f"expected={load_n}, "
                f"actual={len(group_chunks)}."
            )

        output_chunks = []

        aux_chunks = {
            'objectness_logit': [],
            'object_consensus': [],
            'background_consensus': [],
            'dual_attention': [],
        }

        group_sizes = []

        for group_x4, group_eva in zip(
            group_chunks,
            eva_group_chunks,
        ):
            group_output, group_aux = (
                self._run_consensus_once(
                    group_x4,
                    eva_feature_map=group_eva,
                )
            )

            output_chunks.append(group_output)
            group_sizes.append(
                int(group_output.shape[0])
            )

            for key in aux_chunks:
                value = group_aux.get(key)

                if value is not None:
                    aux_chunks[key].append(value)

        merged_aux = {}

        for key, values in aux_chunks.items():
            if values:
                merged_aux[key] = torch.cat(
                    values,
                    dim=0,
                )

        self._last_group_sizes = tuple(group_sizes)

        return (
            torch.cat(output_chunks, dim=0),
            merged_aux,
        )

    def forward(self, x):
        eva_feature_map = None

        if getattr(
            self.config,
            'use_eva_semantic_calibration',
            False,
        ):
            eva_runtime = getattr(
                self,
                '_eva_runtime',
                None,
            )

            if eva_runtime is None:
                raise RuntimeError(
                    'EVA runtime is not initialized.'
                )

            eva_feature_map = eva_runtime.extract(x)

        ########## Encoder ##########

        if 'trans' in self.config.bb:
            x1, x2, x3, x4 = self.bb(x)
        else:
            x1 = self.bb.conv1(x)
            x2 = self.bb.conv2(x1)
            x3 = self.bb.conv3(x2)
            x4 = self.bb.conv4(x3)

        # B0-G: isolate semantic groups during training while
        # preserving whole-group processing during evaluation.
        x4, self._last_aux = (
            self._apply_consensus_pipeline(
                x4,
                eva_feature_map=eva_feature_map,
            )
        )

        if self.lvsp is not None:
            dino_features = getattr(
                self.bb,
                'last_dino_features',
                None,
            )

            if (
                dino_features is None
                or len(dino_features) != 4
            ):
                raise RuntimeError(
                    'LVSP requires DINO features '
                    'from Blocks 2, 5, 8 and 11.'
                )

            dual_attention = (
                self._last_aux.get(
                    'dual_attention',
                    None,
                )
            )

            if dual_attention is None:
                raise RuntimeError(
                    'LVSP requires OV-DCD '
                    'dual_attention.'
                )

            x3 = self.lvsp(
                x3=x3,
                x4=x4,
                f8=dino_features[2],
                f11=dino_features[3],
                dual_attention=dual_attention,
            )

            self._last_aux.update({
                'lvsp_variation':
                    self.lvsp.last_variation,
                'lvsp_uncertainty':
                    self.lvsp.last_uncertainty,
                'lvsp_gate':
                    self.lvsp.last_gate,
                'lvsp_gamma':
                    self.lvsp.last_gamma,
            })

        ########## Decoder ##########
        scaled_preds = []

        if self.use_dpt_decoder:
            # x4 has already passed through the unchanged
            # DPCM/OCCR/OV-DCD consensus pipeline.
            if self.cgpf is not None:
                required_cgpf_aux = (
                    'object_consensus',
                    'background_consensus',
                    'dual_attention',
                )
                missing_cgpf_aux = [
                    name
                    for name in required_cgpf_aux
                    if (
                        name not in self._last_aux
                        or self._last_aux[name] is None
                    )
                ]
                if missing_cgpf_aux:
                    raise RuntimeError(
                        'CGPF requires merged OV-DCD '
                        'auxiliary maps, missing: {}'.format(
                            missing_cgpf_aux
                        )
                    )
                x3 = self.cgpf(
                    x3=x3,
                    x4=x4,
                    object_consensus=(
                        self._last_aux['object_consensus']
                    ),
                    background_consensus=(
                        self._last_aux['background_consensus']
                    ),
                    dual_attention=(
                        self._last_aux['dual_attention']
                    ),
                    group_sizes=getattr(
                        self,
                        '_last_group_sizes',
                        None,
                    ),
                )

            p1_out = self.dpt_decoder(
                x1=x1,
                x2=x2,
                x3=x3,
                x4=x4,
                output_size=x.shape[-2:],
            )
        else:
            p4 = self.top_layer(x4)
            p4 = self.dec_layer4(p4)
            p4 = F.interpolate(
                p4,
                size=x3.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

            if getattr(
                self.config,
                'use_x3_objectness_guidance',
                False,
            ):
                shared_objectness_logit = (
                    self._last_aux.get(
                        'objectness_logit',
                        None,
                    )
                )

                if shared_objectness_logit is None:
                    raise RuntimeError(
                        "x3 guidance requires the shared "
                        "objectness logit."
                    )

                objectness_x3 = torch.sigmoid(
                    F.interpolate(
                        shared_objectness_logit,
                        size=x3.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                )

                x3_gate = (
                    1.0
                    + self.x3_guidance_strength
                    * (
                        2.0 * objectness_x3
                        - 1.0
                    )
                )

                x3_for_decoder = x3 * x3_gate
                self._last_x3_objectness_gate = (
                    x3_gate.detach()
                )
            else:
                x3_for_decoder = x3
                self._last_x3_objectness_gate = None

            p3 = (
                p4
                + self.lat_layer4(
                    x3_for_decoder
                )
            )

            p3 = self.dec_layer3(p3)
            p3 = F.interpolate(
                p3,
                size=x2.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

            p2 = p3 + self.lat_layer3(x2)
            p2 = self.dec_layer2(p2)
            p2 = F.interpolate(
                p2,
                size=x1.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

            p1 = p2 + self.lat_layer2(x1)
            p1 = self.dec_layer1(p1)
            p1 = F.interpolate(
                p1,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

            if self.config.db_output_decoder:
                p1_out = self.db_output_decoder(p1)
            else:
                p1_out = self.conv_out1(p1)

        scaled_preds.append(p1_out)

        if self.training:
            return_values = [scaled_preds, x4]
            return return_values
        else:
            return scaled_preds
