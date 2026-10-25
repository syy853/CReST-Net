from random import seed
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from util import Logger, AverageMeter, set_seed
import os
import argparse
from dataset import get_loader

import torch.nn.functional as F

from config import Config
from loss import saliency_structure_consistency, SalLoss
# OHR-Net: CCRD removed. No CLIP / no text bank.
# OHR-Net: SGIL removed. H-PRI is used instead.
from dino_teacher_loss import DINOTeacherObjectnessLoss
from optimizer_utils import (
    build_optimizer,
    validate_optimizer_contract,
)
from util import generate_smoothed_gt

from models.GCoNet import GCoNet


# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='GCoNet',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--smoke_batches', type=int, default=0, help='stop training after N batches for smoke test')
parser.add_argument('--start_epoch',
                    default=1,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='DUTS_class+coco-seg',
                    type=str,
                    help="Options: 'DUTS_class'")
parser.add_argument('--size',
                    default=256,
                    type=int,
                    help='input size')
parser.add_argument('--ckpt_dir', default=None, help='Temporary folder')

parser.add_argument('--testsets',
                    default='CoCA+CoSOD3k+CoSal2015',
                    type=str,
                    help="Options: 'CoCA', 'CoSal2015', 'CoSOD3k'")

args = parser.parse_args()


config = Config()

# Prepare dataset
root_dir = os.path.join(config.proj_root, 'data')
if 'DUTS_class' in args.trainset.split('+'):
    train_img_path = os.path.join(root_dir, 'images/DUTS_class')
    train_gt_path = os.path.join(root_dir, 'gts/DUTS_class')
    train_loader = get_loader(
        train_img_path,
        train_gt_path,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )
if 'coco-seg' in args.trainset.split('+'):
    train_img_path_seg = os.path.join(root_dir, 'images/coco-seg')
    train_gt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    train_loader_seg = get_loader(
        train_img_path_seg,
        train_gt_path_seg,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )
if 'coco-9k' in args.trainset.split('+'):
    train_img_path_seg = os.path.join(root_dir, 'images/coco-9k')
    train_gt_path_seg = os.path.join(root_dir, 'gts/coco-9k')
    train_loader_seg = get_loader(
        train_img_path_seg,
        train_gt_path_seg,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )
# else:
#     print('Unkonwn train dataset')
#     print(args.dataset)

test_loaders = {}
for testset in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join(root_dir, 'images', testset), os.path.join(root_dir, 'gts', testset),
        args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True
    )
    test_loaders[testset] = test_loader

if config.rand_seed:
    set_seed(config.rand_seed)

# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_file = os.path.join(args.ckpt_dir, "log_loss.txt")
logger_loss_idx = 1

# Init model
device = torch.device("cuda")

model = GCoNet().to(device)

# DINOv2 training contract:
#   frozen foundation backbone;
#   trainable pyramid adapters and E6 task modules;
#   optimizer contains only requires_grad=True parameters.
optimizer = build_optimizer(
    model,
    config,
)

optimizer_contract = validate_optimizer_contract(
    model,
    optimizer,
)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        lde
        if lde > 0
        else args.epochs + lde
        for lde in config.lr_decay_epochs
    ],
    gamma=0.1,
)

if config.lambda_adv_g:
    from adv import Discriminator
    disc = Discriminator(channels=3, img_size=args.size).to(device)
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()
    if config.optimizer == 'AdamW':
        optimizer_d = optim.AdamW(params=disc.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_d,
        milestones=[lde if lde > 0 else args.epochs + lde for lde in config.lr_decay_epochs],
        gamma=0.1
    )

# The legacy whole-bb freeze has been removed.
# Freezing after optimizer construction would incorrectly disable
# the trainable DINOv2 pyramid adapters.
if getattr(config, 'freeze', False):
    raise RuntimeError(
        "Legacy config.freeze must remain False for the "
        "DINOv2 Pyramid Adapter branch."
    )


# log model and optimizer params
# logger.info("Model details:")
# logger.info(model)
logger.info("Optimizer contract: {}".format(optimizer_contract))
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(lr_scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
sal_loss = SalLoss()

dino_teacher_criterion = None
if getattr(config, 'use_dino_teacher', False):
    dino_teacher_criterion = DINOTeacherObjectnessLoss(
        model_name=getattr(config, 'dino_teacher_model', 'vit_small_patch14_dinov2'),
        ckpt_path=getattr(config, 'dino_teacher_ckpt', 'pretrained/dinov2_vits14_pretrain.pth'),
        input_size=getattr(config, 'dino_teacher_input_size', 518),
        temperature=getattr(config, 'dino_teacher_temperature', 0.07),
    ).cuda()
    logging.info(
        'OHR-Net DINO teacher enabled: model=%s, ckpt=%s, loaded_keys=%s',
        getattr(config, 'dino_teacher_model', 'vit_small_patch14_dinov2'),
        getattr(config, 'dino_teacher_ckpt', ''),
        getattr(dino_teacher_criterion, 'loaded_keys', 'NA')
    )

ccrd_loss = None  # OHR-Net strict No-CLIP: CCRD disabled
sgil_loss = None  # OHR-Net: SGIL replaced by unified H-PRI
logging.info('OHR-Net v3 setup: use_ccrd=%s, use_sgil=%s, use_hpri=%s, GCAM_metric=%s',
             getattr(config, 'use_ccrd', None),
             getattr(config, 'use_sgil', None),
             getattr(config, 'use_hpri', None),
             getattr(config, 'GCAM_metric', None))


def get_model_aux_outputs(model):
    module = model.module if hasattr(model, 'module') else model

    if hasattr(module, 'get_aux_outputs'):
        return module.get_aux_outputs()

    return {}


def gt_guided_objectness_loss(model, gt):
    """
    One shared objectness supervision serves both candidate-level
    OCCR and pixel-level OV-DCD. No second pre-objectness loss exists.
    """
    aux = get_model_aux_outputs(model)

    objectness_logit = aux.get(
        'objectness_logit',
        None,
    )

    if objectness_logit is None:
        return None

    gt_small = F.interpolate(
        gt.float(),
        size=objectness_logit.shape[-2:],
        mode='nearest',
    )

    return F.binary_cross_entropy_with_logits(
        objectness_logit,
        gt_small,
    )


def main():
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs+1):
        train_loss = train(epoch)
        # Save checkpoint
        if epoch >= args.epochs - config.val_last and (args.epochs - epoch) % config.save_step == 0:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch)))
        lr_scheduler.step()
        if config.lambda_adv_g:
            lr_scheduler_d.step()


def train(epoch):
    loss_log = AverageMeter()
    loss_log_metric = AverageMeter()
    global logger_loss_idx
    model.train()

    for batch_idx, (batch, batch_seg) in enumerate(zip(train_loader, train_loader_seg)):
        if getattr(args, 'smoke_batches', 0) and batch_idx >= args.smoke_batches:
            print(f"[SMOKE_STOP] reached smoke_batches={args.smoke_batches}")
            break
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        cls_gts = (
            torch.LongTensor(batch[-1])
            .to(device)
            .reshape(-1)
        )

        return_values = model(inputs)
        scaled_preds = return_values[0]
        norm_features = None
        if config.GCAM_metric:
            norm_features = return_values[-1]
        scaled_preds = scaled_preds[-min(config.loss_sal_layers+int(bool(config.refine)), 4+int(bool(config.refine))):]

        # Tricks
        if config.GCAM_metric:
            loss_sal, loss_metric = sal_loss(scaled_preds, gts, norm_features=norm_features, labels=cls_gts)

            loss_ccrd = None

            if getattr(config, 'use_ccrd', False) and epoch >= getattr(config, 'ccrd_start_epoch', 1):

                loss_ccrd = ccrd_loss(norm_features, scaled_preds[-1], cls_gts, dataset_name='DUTS_class')

                if loss_ccrd is not None:

                    loss_sal = loss_sal + loss_ccrd * getattr(config, 'lambda_ccrd', 0.01)

                    loss_metric = loss_metric + loss_ccrd.detach() * getattr(config, 'lambda_ccrd', 0.01)
            loss_sgil = None
            if getattr(config, 'use_sgil', False) and epoch >= getattr(config, 'sgil_start_epoch', 1):
                loss_sgil = sgil_loss(norm_features, scaled_preds[-1], gts, cls_gts, dataset_name='DUTS_class')
                if loss_sgil is not None:
                    loss_sal = loss_sal + loss_sgil * getattr(config, 'lambda_sgil', 0.05)
                    loss_metric = loss_metric + loss_sgil.detach() * getattr(config, 'lambda_sgil', 0.05)

            loss_objectness = None
            if getattr(config, 'use_gt_objectness', False) and epoch >= getattr(config, 'objectness_start_epoch', 1):
                loss_objectness = gt_guided_objectness_loss(model, gts)
                if loss_objectness is not None:
                    loss_sal = loss_sal + loss_objectness * getattr(config, 'lambda_objectness', 0.05)

            loss_dino_teacher = None
            if dino_teacher_criterion is not None and epoch >= getattr(config, 'dino_teacher_start_epoch', 1):
                loss_dino_teacher = dino_teacher_criterion(scaled_preds[-1], inputs, gts)
                loss_sal = loss_sal + loss_dino_teacher * getattr(config, 'lambda_dino_teacher', 0.02)
        else:
            loss_sal = sal_loss(scaled_preds, gts)
        if config.label_smoothing:
            loss_sal = 0.5 * (loss_sal + sal_loss(scaled_preds, generate_smoothed_gt(gts)))
        if config.self_supervision:
            H, W = inputs.shape[-2:]
            images_scale = F.interpolate(inputs, size=(H//4, W//4), mode='bilinear', align_corners=True)
            sal_scale = model(images_scale)[0][-1]
            atts = scaled_preds[-1]
            sal_s = F.interpolate(atts, size=(H//4, W//4), mode='bilinear', align_corners=True)
            loss_ss = saliency_structure_consistency(sal_scale.sigmoid(), sal_s.sigmoid())
            loss_sal += loss_ss * 0.3

        # Loss
        # since there may be several losses for sal, the lambdas for them (lambdas_sal) are inside the loss.py
        loss = loss_sal * 1.0
        if config.lambda_adv_g:
            # gen
            valid = Variable(Tensor(scaled_preds[-1].shape[0], 1).fill_(1.0), requires_grad=False)
            adv_loss_g = adv_criterion(disc(scaled_preds[-1] * inputs), valid)
            loss += adv_loss_g * config.lambda_adv_g

        if config.forward_per_dataset:
            loss_log.update(loss, inputs.size(0))
            if config.GCAM_metric:
                loss_log_metric.update(loss_metric, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
        inputs = batch_seg[0].to(device).squeeze(0)
        gts = batch_seg[1].to(device).squeeze(0)
        cls_gts = (
            torch.LongTensor(batch_seg[-1])
            .to(device)
            .reshape(-1)
        )

        return_values = model(inputs)
        scaled_preds = return_values[0]
        norm_features = None
        if config.GCAM_metric:
            norm_features = return_values[-1]
        scaled_preds = scaled_preds[-min(config.loss_sal_layers+int(bool(config.refine)), 4+int(bool(config.refine))):]

        # Tricks
        if config.GCAM_metric:
            loss_sal, loss_metric = sal_loss(scaled_preds, gts, norm_features=norm_features, labels=cls_gts)

            loss_ccrd = None

            if getattr(config, 'use_ccrd', False) and epoch >= getattr(config, 'ccrd_start_epoch', 1):

                loss_ccrd = ccrd_loss(norm_features, scaled_preds[-1], cls_gts, dataset_name='coco-seg')

                if loss_ccrd is not None:

                    loss_sal = loss_sal + loss_ccrd * getattr(config, 'lambda_ccrd', 0.01)

                    loss_metric = loss_metric + loss_ccrd.detach() * getattr(config, 'lambda_ccrd', 0.01)
            loss_sgil = None
            if getattr(config, 'use_sgil', False) and epoch >= getattr(config, 'sgil_start_epoch', 1):
                loss_sgil = sgil_loss(norm_features, scaled_preds[-1], gts, cls_gts, dataset_name='coco-seg')
                if loss_sgil is not None:
                    loss_sal = loss_sal + loss_sgil * getattr(config, 'lambda_sgil', 0.05)
                    loss_metric = loss_metric + loss_sgil.detach() * getattr(config, 'lambda_sgil', 0.05)

            loss_objectness = None
            if getattr(config, 'use_gt_objectness', False) and epoch >= getattr(config, 'objectness_start_epoch', 1):
                loss_objectness = gt_guided_objectness_loss(model, gts)
                if loss_objectness is not None:
                    loss_sal = loss_sal + loss_objectness * getattr(config, 'lambda_objectness', 0.05)

            loss_dino_teacher = None
            if dino_teacher_criterion is not None and epoch >= getattr(config, 'dino_teacher_start_epoch', 1):
                loss_dino_teacher = dino_teacher_criterion(scaled_preds[-1], inputs, gts)
                loss_sal = loss_sal + loss_dino_teacher * getattr(config, 'lambda_dino_teacher', 0.02)
        else:
            loss_sal = sal_loss(scaled_preds, gts)
        if config.label_smoothing:
            loss_sal = 0.5 * (loss_sal + sal_loss(scaled_preds, generate_smoothed_gt(gts)))
        if config.self_supervision:
            H, W = inputs.shape[-2:]
            images_scale = F.interpolate(inputs, size=(H//4, W//4), mode='bilinear', align_corners=True)
            sal_scale = model(images_scale)[0][-1]
            atts = scaled_preds[-1]
            sal_s = F.interpolate(atts, size=(H//4, W//4), mode='bilinear', align_corners=True)
            loss_ss = saliency_structure_consistency(sal_scale.sigmoid(), sal_s.sigmoid())
            loss_sal += loss_ss * 0.3

        # Loss
        if config.forward_per_dataset:
            loss = loss_sal * 1.0
        else:
            loss += loss_sal * 1.0

        if config.lambda_adv_g:
            # gen
            valid = Variable(Tensor(scaled_preds[-1].shape[0], 1).fill_(1.0), requires_grad=False)
            adv_loss_g = adv_criterion(disc(scaled_preds[-1] * inputs), valid)
            loss += adv_loss_g * config.lambda_adv_g

        loss_log.update(loss, inputs.size(0))
        if config.GCAM_metric:
            loss_log_metric.update(loss_metric, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with open(logger_loss_file, 'a') as f:
            f.write('step {}, {}\n'.format(logger_loss_idx, loss))
        logger_loss_idx += 1
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

        if config.lambda_adv_g and batch_idx % 2 == 0:
            # disc
            fake = Variable(Tensor(scaled_preds[-1].shape[0], 1).fill_(0.0), requires_grad=False)
            optimizer_d.zero_grad()
            adv_loss_real = adv_criterion(disc(gts * inputs), valid)
            adv_loss_fake = adv_criterion(disc(scaled_preds[-1].detach() * inputs.detach()), fake)
            adv_loss_d = (adv_loss_real + adv_loss_fake) / 2 * config.lambda_adv_d
            adv_loss_d.backward()
            optimizer_d.step()

        # Logger
        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}]'.format(epoch, args.epochs, batch_idx, len(train_loader))
            info_loss = 'Train Loss: loss_sal: {:.3f}'.format(loss_sal)
            if 'loss_objectness' in locals() and loss_objectness is not None:
                info_loss += ' loss_objectness: {:.4f}'.format(float(loss_objectness.detach().cpu()))
            if 'loss_dino_teacher' in locals() and loss_dino_teacher is not None:
                info_loss += ' loss_dino_teacher: {:.4f}'.format(float(loss_dino_teacher.detach().cpu()))
            if config.lambda_adv_g:
                info_loss += ', loss_adv: {:.3f}, loss_adv_disc: {:.3f}'.format(adv_loss_g * config.lambda_adv_g, adv_loss_d * config.lambda_adv_d)
            if config.GCAM_metric:
                info_loss += ', loss_hpri: {:.3f}'.format(loss_metric)
            info_loss += ', Loss_total: {loss.val:.3f} ({loss.avg:.3f})  '.format(loss=loss_log)
            logger.info(''.join((info_progress, info_loss)))
    info_loss = '@==Final== Epoch[{0}/{1}]  Train Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=loss_log)
    if config.GCAM_metric:
        info_loss += 'HPRI Loss: {loss.avg:.3f}  '.format(loss=loss_log_metric)
    logger.info(info_loss)

    return loss_log.avg


if __name__ == '__main__':
    main()
