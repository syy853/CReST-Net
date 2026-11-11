import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

config = Config()

class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, groups=0):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        if config.dec_att == 'ASPP':
            self.dec_att = ASPP(channel_in=64)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(64)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        if config.dec_att:
            x = self.dec_att(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x


class _ASPPModule(nn.Module):
    def __init__(self, channel_in, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(channel_in, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, channel_in=64, output_stride=16):
        super(ASPP, self).__init__()
        self.down_scale = 1
        self.channel_inter = 256 // self.down_scale
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(channel_in, self.channel_inter, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(channel_in, self.channel_inter, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(channel_in, self.channel_inter, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.channel_inter),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.channel_inter * 5, channel_in, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_in)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)











# -------------------------------------------------------------------------
# OHR-Net v2: Objectness-Verified Dual Consensus Decomposition
# -------------------------------------------------------------------------
class ObjectnessVerifiedDCD(nn.Module):
    """
    Shared-Objectness Dual Consensus Decomposition.

    This module no longer predicts a second objectness map. It receives
    the shared objectness logit already used by OCCR, then reuses the
    same prior to construct object/background consensus prototypes from
    the calibrated DPCM feature.
    """

    def __init__(
        self,
        channels,
        alpha=0.7,
        temperature=4.0,
        gamma_init=0.0,
    ):
        super(ObjectnessVerifiedDCD, self).__init__()

        self.alpha = float(alpha)
        self.temperature = float(temperature)

        def _gn(ch, preferred):
            for group in [
                preferred,
                32,
                16,
                8,
                4,
                2,
                1,
            ]:
                if group <= ch and ch % group == 0:
                    return nn.GroupNorm(group, ch)

            return nn.GroupNorm(1, ch)

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            _gn(channels, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init))
        )

        self.last_objectness_logit = None
        self.last_object_consensus = None
        self.last_background_consensus = None
        self.last_dual_attention = None

    @staticmethod
    def _weighted_pool(
        feat,
        weight,
        eps=1e-6,
    ):
        # feat: B,C,H,W; weight: B,1,H,W
        denominator = weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(eps)

        return (
            feat * weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        ) / denominator

    def forward(
        self,
        feat,
        shared_objectness_logit,
    ):
        if shared_objectness_logit is None:
            raise RuntimeError(
                "Shared-objectness OV-DCD requires "
                "an objectness logit."
            )

        if shared_objectness_logit.dim() != 4:
            raise ValueError(
                "shared_objectness_logit must be "
                "B,1,H,W, got: {}".format(
                    tuple(
                        shared_objectness_logit.shape
                    )
                )
            )

        if shared_objectness_logit.shape[1] != 1:
            raise ValueError(
                "Shared objectness must have one "
                "channel, got: {}".format(
                    shared_objectness_logit.shape[1]
                )
            )

        if (
            shared_objectness_logit.shape[0]
            != feat.shape[0]
        ):
            raise ValueError(
                "Feature/objectness batch mismatch: "
                "{} vs {}".format(
                    feat.shape[0],
                    shared_objectness_logit.shape[0],
                )
            )

        if (
            shared_objectness_logit.shape[-2:]
            != feat.shape[-2:]
        ):
            shared_objectness_logit = F.interpolate(
                shared_objectness_logit,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        objectness_prob = torch.sigmoid(
            shared_objectness_logit
        )

        object_proto_image = self._weighted_pool(
            feat,
            objectness_prob,
        )

        background_proto_image = self._weighted_pool(
            feat,
            1.0 - objectness_prob,
        )

        # Current batch is one isolated semantic group.
        object_proto = object_proto_image.mean(
            dim=0,
            keepdim=True,
        )

        background_proto = (
            background_proto_image.mean(
                dim=0,
                keepdim=True,
            )
        )

        feat_normalized = F.normalize(
            feat,
            dim=1,
        )

        object_normalized = F.normalize(
            object_proto,
            dim=1,
        )

        background_normalized = F.normalize(
            background_proto,
            dim=1,
        )

        object_similarity = (
            feat_normalized
            * object_normalized
        ).sum(
            dim=1,
            keepdim=True,
        )

        background_similarity = (
            feat_normalized
            * background_normalized
        ).sum(
            dim=1,
            keepdim=True,
        )

        dual_score = (
            object_similarity
            - self.alpha
            * background_similarity
        )

        dual_attention = torch.sigmoid(
            self.temperature
            * dual_score
        )

        refined = (
            feat
            + self.gamma
            * self.refine(
                feat * dual_attention
            )
        )

        self.last_objectness_logit = (
            shared_objectness_logit.detach()
        )

        self.last_object_consensus = (
            object_similarity.detach()
        )

        self.last_background_consensus = (
            background_similarity.detach()
        )

        self.last_dual_attention = (
            dual_attention.detach()
        )

        return refined


class DiversePrototypeConsensusMining(nn.Module):
    """
    Diverse Prototype Consensus Mining (DPCM).

    DPCM mines multiple candidate object-consensus prototypes from high-level
    semantic features. It does not perform positive/background decomposition.
    That decomposition is handled by OV-DCD.

    Module responsibility:
        x4 -> multiple candidate object-consensus prototypes -> x4_c

    Difference from OV-DCD:
        DPCM mines diverse candidate object consensus.
        OV-DCD decomposes the candidate consensus into object/background consensus.
    """

    def __init__(
        self,
        channel_in,
        num_prototypes=4,
        temperature=4.0,
        gamma_init=0.1,
        reduction=4,
        routing_mode='hard',
        routing_temperature=0.1,
        use_shared_objectness=False,
        use_occr=False,
        occr_alpha_raw_init=-6.0,
        occr_alpha_max=2.0,
        occr_eps=1e-6,
        use_eva_semantic=False,
        eva_semantic_raw_delta_init=0.0,
        eva_semantic_delta_max=1.0,
        eva_semantic_eps=1e-6,
    ):
        super(DiversePrototypeConsensusMining, self).__init__()

        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)

        self.routing_mode = str(routing_mode).lower()
        self.routing_temperature = float(routing_temperature)

        self.use_shared_objectness = bool(
            use_shared_objectness
        )
        self.use_occr = bool(use_occr)

        if (
            self.use_occr
            and not self.use_shared_objectness
        ):
            raise ValueError(
                "OCCR requires use_shared_objectness=True."
            )

        self.occr_alpha_max = float(occr_alpha_max)
        self.occr_eps = float(occr_eps)

        self.use_eva_semantic = bool(use_eva_semantic)
        self.eva_semantic_delta_max = float(
            eva_semantic_delta_max
        )
        self.eva_semantic_eps = float(eva_semantic_eps)

        if self.eva_semantic_delta_max <= 0.0:
            raise ValueError(
                'eva_semantic_delta_max must be positive, got: {}'.format(
                    eva_semantic_delta_max
                )
            )

        if self.eva_semantic_eps <= 0.0:
            raise ValueError(
                'eva_semantic_eps must be positive, got: {}'.format(
                    eva_semantic_eps
                )
            )

        if self.use_eva_semantic:
            self.eva_semantic_raw_delta = nn.Parameter(
                torch.tensor(float(eva_semantic_raw_delta_init))
            )
        else:
            self.register_parameter(
                'eva_semantic_raw_delta',
                None,
            )

        if self.occr_alpha_max <= 0.0:
            raise ValueError(
                'occr_alpha_max must be positive, got: {}'.format(
                    occr_alpha_max
                )
            )

        if self.occr_eps <= 0.0:
            raise ValueError(
                'occr_eps must be positive, got: {}'.format(
                    occr_eps
                )
            )

        if self.routing_mode not in ('hard', 'soft'):
            raise ValueError(
                'routing_mode must be hard or soft, got: {}'.format(
                    routing_mode
                )
            )

        if self.routing_temperature <= 0.0:
            raise ValueError(
                'routing_temperature must be positive, got: {}'.format(
                    routing_temperature
                )
            )

        hidden = max(channel_in // reduction, 32)

        def _gn(ch, preferred):
            for g in [preferred, 32, 16, 8, 4, 2, 1]:
                if g <= ch and ch % g == 0:
                    return nn.GroupNorm(g, ch)
            return nn.GroupNorm(1, ch)

        # Shared Objectness predictor:
        # Predict coarse objectness from the original x4 feature before
        # candidate consensus mining. The predictor may be used by:
        # 1) OCCR candidate-level routing calibration;
        # 2) OV-DCD pixel-level object/background verification.
        if self.use_shared_objectness:
            self.shared_objectness_head = nn.Sequential(
                nn.Conv2d(
                    channel_in,
                    hidden,
                    kernel_size=1,
                    bias=False,
                ),
                _gn(hidden, 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden,
                    1,
                    kernel_size=1,
                ),
            )
        else:
            self.shared_objectness_head = None

        # OCCR-only learnable calibration strength:
        # softplus(occr_alpha_raw), bounded by occr_alpha_max.
        # This parameter must not exist when OCCR is disabled.
        if self.use_occr:
            self.occr_alpha_raw = nn.Parameter(
                torch.tensor(
                    float(occr_alpha_raw_init)
                )
            )
        else:
            self.register_parameter(
                'occr_alpha_raw',
                None,
            )

        # Generate K candidate response maps.
        self.score_head = nn.Sequential(
            nn.Conv2d(channel_in, hidden, kernel_size=1, bias=False),
            _gn(hidden, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.num_prototypes, kernel_size=1)
        )

        # Residual refinement after candidate consensus attention.
        self.refine = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=1, bias=False),
            _gn(channel_in, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, bias=False)
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # Saved only for later visualization/debugging.
        self.last_seed_maps = None
        self.last_consensus_attention = None
        self.last_group_prototypes = None
        self.last_routing_weights = None
        self.last_routed_similarity = None

        # E2-O / OCCR diagnostics and auxiliary supervision.
        self.last_shared_objectness_logit = None
        self.last_candidate_objectness_support = None
        self.last_occr_candidate_bias = None
        self.last_occr_alpha = None
        self.last_routing_logits = None
        self.last_eva_semantic_agreement = None
        self.last_eva_semantic_bias = None
        self.last_eva_semantic_delta = None

    @staticmethod
    def _candidate_objectness_support(
        seed_weights,
        objectness_prob,
    ):
        """
        Pool shared objectness probability with each spatially normalized
        candidate seed map.

        seed_weights:
            B,K,H,W, sum over H,W equals one for every B,K.

        objectness_prob:
            B,1,H,W.

        returns:
            K candidate-level group objectness supports.
        """
        if seed_weights.dim() != 4:
            raise ValueError(
                'seed_weights must be B,K,H,W, got: {}'.format(
                    tuple(seed_weights.shape)
                )
            )

        if objectness_prob.dim() != 4:
            raise ValueError(
                'objectness_prob must be B,1,H,W, got: {}'.format(
                    tuple(objectness_prob.shape)
                )
            )

        if objectness_prob.shape[1] != 1:
            raise ValueError(
                'objectness_prob channel must be one, got: {}'.format(
                    objectness_prob.shape[1]
                )
            )

        if (
            seed_weights.shape[0] != objectness_prob.shape[0]
            or seed_weights.shape[-2:] != objectness_prob.shape[-2:]
        ):
            raise ValueError(
                'seed/objectness shape mismatch: {} vs {}'.format(
                    tuple(seed_weights.shape),
                    tuple(objectness_prob.shape),
                )
            )

        support_per_image = (
            seed_weights
            * objectness_prob
        ).sum(dim=(2, 3))

        return support_per_image.mean(dim=0)

    def _candidate_eva_semantic_bias(
        self,
        seed_weights,
        eva_feature_map,
    ):
        """
        Compute candidate-wise semantic agreement inside the current
        isolated semantic group.

        seed_weights: B,K,H,W.
        eva_feature_map: B,C_e,H_e,W_e.

        returns:
            agreement: K
            centered_bias: K
        """
        if eva_feature_map is None:
            raise RuntimeError(
                'EVA semantic calibration requires an EVA feature map.'
            )

        if eva_feature_map.dim() != 4:
            raise ValueError(
                'eva_feature_map must be B,C,H,W, got: {}'.format(
                    tuple(eva_feature_map.shape)
                )
            )

        if eva_feature_map.shape[0] != seed_weights.shape[0]:
            raise ValueError(
                'Seed/EVA batch mismatch: {} vs {}'.format(
                    seed_weights.shape[0],
                    eva_feature_map.shape[0],
                )
            )

        resized_seed = F.interpolate(
            seed_weights,
            size=eva_feature_map.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).clamp_min(0.0)

        resized_seed = resized_seed / (
            resized_seed.sum(
                dim=(2, 3),
                keepdim=True,
            ).clamp_min(self.eva_semantic_eps)
        )

        candidate_proto = torch.einsum(
            'bkhw,bchw->bkc',
            resized_seed,
            eva_feature_map,
        )

        candidate_proto = F.normalize(
            candidate_proto,
            dim=2,
            eps=self.eva_semantic_eps,
        )

        group_proto = F.normalize(
            candidate_proto.mean(dim=0),
            dim=1,
            eps=self.eva_semantic_eps,
        )

        agreement = (
            candidate_proto
            * group_proto.unsqueeze(0)
        ).sum(dim=2).mean(dim=0)

        centered_bias = agreement - agreement.mean()

        return agreement, centered_bias

    def _soft_route(
        self,
        sim,
        candidate_objectness_support=None,
        eva_semantic_bias=None,
    ):
        routing_logits = sim / self.routing_temperature

        candidate_bias = sim.new_zeros(
            self.num_prototypes
        )
        occr_alpha = sim.new_zeros(())
        eva_semantic_delta = sim.new_zeros(())

        if self.use_occr:
            if candidate_objectness_support is None:
                raise RuntimeError(
                    'OCCR requires candidate objectness support.'
                )

            if candidate_objectness_support.shape != (
                self.num_prototypes,
            ):
                raise RuntimeError(
                    'Unexpected candidate objectness support shape: '
                    '{}'.format(
                        tuple(
                            candidate_objectness_support.shape
                        )
                    )
                )

            support_log = torch.log(
                candidate_objectness_support.clamp_min(
                    self.occr_eps
                )
            )

            candidate_bias = (
                support_log
                - support_log.mean()
            )

            occr_alpha = F.softplus(
                self.occr_alpha_raw
            ).clamp(
                max=self.occr_alpha_max
            )

            routing_logits = (
                routing_logits
                + occr_alpha
                * candidate_bias.view(1, -1, 1, 1)
            )

        if self.use_eva_semantic:
            if eva_semantic_bias is None:
                raise RuntimeError(
                    'EVA semantic routing requires candidate bias.'
                )

            if eva_semantic_bias.shape != (
                self.num_prototypes,
            ):
                raise RuntimeError(
                    'Unexpected EVA semantic bias shape: {}'.format(
                        tuple(eva_semantic_bias.shape)
                    )
                )

            eva_semantic_delta = (
                self.eva_semantic_delta_max
                * torch.tanh(self.eva_semantic_raw_delta)
            )

            routing_logits = (
                routing_logits
                + eva_semantic_delta
                * eva_semantic_bias.view(1, -1, 1, 1)
            )

        routing_weights = F.softmax(
            routing_logits,
            dim=1,
        )

        routed_sim = torch.sum(
            routing_weights * sim,
            dim=1,
            keepdim=True,
        )

        return (
            routing_weights,
            routed_sim,
            routing_logits,
            candidate_bias,
            occr_alpha,
            eva_semantic_delta,
        )

    def forward(
        self,
        x,
        eva_feature_map=None,
    ):
        b, c, h, w = x.shape
        k = self.num_prototypes

        # K candidate response maps: B,K,H,W.
        seed_logits = self.score_head(x)
        seed_weights = F.softmax(
            seed_logits.view(b, k, -1),
            dim=-1,
        ).view(b, k, h, w)

        pre_objectness_logit = None
        candidate_objectness_support = None
        eva_semantic_agreement = None
        eva_semantic_bias = None
        eva_semantic_delta = x.new_zeros(())

        if self.use_shared_objectness:
            if self.shared_objectness_head is None:
                raise RuntimeError(
                    "Shared Objectness Head is not initialized."
                )

            pre_objectness_logit = (
                self.shared_objectness_head(x)
            )

        if self.use_occr:
            if pre_objectness_logit is None:
                raise RuntimeError(
                    "OCCR requires a shared objectness logit."
                )

            pre_objectness_prob = torch.sigmoid(
                pre_objectness_logit
            )

            candidate_objectness_support = (
                self._candidate_objectness_support(
                    seed_weights,
                    pre_objectness_prob,
                )
            )

        if self.use_eva_semantic:
            (
                eva_semantic_agreement,
                eva_semantic_bias,
            ) = self._candidate_eva_semantic_bias(
                seed_weights,
                eva_feature_map,
            )

        # Image-level candidate prototypes: B,K,C.
        proto_img = torch.einsum('bkhw,bchw->bkc', seed_weights, x)

        # Group-level candidate prototypes: K,C.
        # The batch is treated as the current image group, consistent with the
        # group-based training protocol.
        proto_group = proto_img.mean(dim=0)

        # Prototype-to-pixel similarity: B,K,H,W.
        x_n = F.normalize(x, dim=1)
        p_n = F.normalize(proto_group, dim=1)
        sim = torch.einsum('bchw,kc->bkhw', x_n, p_n)

        # Candidate routing over the K prototype-to-pixel similarities.
        #
        # hard:
        #     Exact B0-G behavior. Only the strongest candidate receives
        #     gradient at each pixel.
        #
        # soft:
        #     Differentiable candidate routing. All candidates may receive
        #     gradient according to their pixel-wise routing probability.
        routing_logits = None
        candidate_bias = sim.new_zeros(k)
        occr_alpha = sim.new_zeros(())

        if self.routing_mode == 'hard':
            routed_sim, winner_idx = sim.max(
                dim=1,
                keepdim=True,
            )

            # Debug-only one-hot routing map. It does not participate in
            # the hard-mode forward computation.
            routing_weights = torch.zeros_like(sim)
            routing_weights.scatter_(1, winner_idx, 1.0)
        else:
            (
                routing_weights,
                routed_sim,
                routing_logits,
                candidate_bias,
                occr_alpha,
                eva_semantic_delta,
            ) = self._soft_route(
                sim,
                candidate_objectness_support,
                eva_semantic_bias,
            )

        consensus_att = torch.sigmoid(
            self.temperature * routed_sim
        )

        out = x + self.gamma * self.refine(x * consensus_att)

        self.last_seed_maps = seed_weights.detach()
        self.last_consensus_attention = consensus_att.detach()
        self.last_group_prototypes = proto_group.detach()
        self.last_routing_weights = routing_weights.detach()
        self.last_routed_similarity = routed_sim.detach()

        # Keep pre_objectness_logit attached to the graph because train.py
        # consumes it through CReSTNet.get_aux_outputs().
        self.last_shared_objectness_logit = (
            pre_objectness_logit
        )

        self.last_candidate_objectness_support = (
            None
            if candidate_objectness_support is None
            else candidate_objectness_support.detach()
        )

        self.last_occr_candidate_bias = (
            candidate_bias.detach()
        )

        self.last_occr_alpha = (
            occr_alpha.detach()
        )

        self.last_routing_logits = (
            None
            if routing_logits is None
            else routing_logits.detach()
        )

        self.last_eva_semantic_agreement = (
            None
            if eva_semantic_agreement is None
            else eva_semantic_agreement.detach()
        )

        self.last_eva_semantic_bias = (
            None
            if eva_semantic_bias is None
            else eva_semantic_bias.detach()
        )

        self.last_eva_semantic_delta = (
            eva_semantic_delta.detach()
        )

        return out
