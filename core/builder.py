from typing import Callable

import torch
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

from core.consistency_loss import PartialConsistencyLoss

__all__ = ["make_dataset", "make_model", "make_criterion", "make_optimizer", "make_scheduler"]


def make_dataset() -> Dataset:
    if configs.dataset.name == "semantic_kitti":
        from core.datasets import SemanticKITTI

        dataset = SemanticKITTI(
            root=configs.dataset.root,
            num_points=configs.dataset.num_points,
            voxel_size=configs.dataset.voxel_size,
        )
    elif configs.dataset.name == "kitti_360":
        # from core.datasets import KITTI360

        from core.datasets.kitti_360_3d import KITTI360
        from transformers import SegformerFeatureExtractor

        feature_extractor = SegformerFeatureExtractor().from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
        )
        feature_extractor.do_reduce_labels = False
        feature_extractor.size = {"height": 376, "width": 512}

        dataset = KITTI360(
            root=configs.dataset.root,
            num_points=configs.dataset.num_points,
            voxel_size=configs.dataset.voxel_size,
            radius=configs.dataset.radius,
            feature_extractor=feature_extractor,
        )
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


def make_model() -> nn.Module:
    if configs.model.name == "minkunet":
        from core.models.semantic_kitti import MinkUNet

        if "cr" in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        if "input_feat" in configs.model:
            input_feat = configs.model.input_feat
        else:
            input_feat = 4
        model = MinkUNet(num_classes=configs.data.num_classes, cr=cr, input_feat=input_feat)
    elif configs.model.name == "spvcnn":
        from core.models.semantic_kitti import SPVCNN

        if "cr" in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        if "input_feat" in configs.model:
            input_feat = configs.model.input_feat
        else:
            input_feat = 4
        model = SPVCNN(
            num_classes=configs.data.num_classes,
            cr=cr,
            pres=configs.dataset.voxel_size,
            vres=configs.dataset.voxel_size,
            input_feat=input_feat,
        )
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == "cross_entropy":
        # criterion = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)
        criterion = PartialConsistencyLoss(
            nn.CrossEntropyLoss, ignore_index=configs.criterion.ignore_index
        )
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov,
        )
    elif configs.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=configs.optimizer.lr, weight_decay=configs.optimizer.weight_decay
        )
    elif configs.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=configs.optimizer.lr, weight_decay=configs.optimizer.weight_decay
        )
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == "cosine_warmup":
        from functools import partial

        from core.schedulers import cosine_schedule_with_warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                dataset_size=configs.data.training_size,
            ),
        )
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
