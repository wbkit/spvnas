from typing import Any, Dict

import numpy as np
import torch
from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback
from torchmetrics.classification import MulticlassConfusionMatrix
import wandb

from core.evaluation import compute_iou

__all__ = ["MeanIoU"]


class MeanIoU(Callback):
    def __init__(
        self,
        num_classes: int,
        ignore_label: int,
        *,
        output_tensor: str = "outputs",
        target_tensor: str = "targets",
        name: str = "iou",
    ) -> None:
        self.num_classes = num_classes
        self.num_classes = 15
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor

        self.val_mean_cm_student = MulticlassConfusionMatrix(
            num_classes=16,
            normalize="true",
            ignore_index=255,
        ).to("cuda")
        self.step_counter = 0

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)
        self.step_counter = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i in range(self.num_classes):
            self.total_seen[i] += torch.sum(targets == i).item()
            self.total_correct[i] += torch.sum((targets == i) & (outputs == targets)).item()
            self.total_positive[i] += torch.sum(outputs == i).item()

        # 2D
        # mIoU
        outputs_2d = output_dict["outputs_2d"]
        targets_2d = output_dict["targets_2d"]
        self.val_mean_cm_student.update(outputs_2d, targets_2d)
        # Visualise
        if self.step_counter in [0, 99, 200, 300, 400, 500, 600, 700, 810, 900]:
            print("Log images")
            outputs_2d = output_dict["outputs_2d"].argmax(dim=1).detach().cpu().numpy()[0]
            targets_2d = output_dict["targets_2d"].detach().cpu().numpy()[0]
            image_2d = output_dict["images"].detach().cpu().numpy()[0]
            image_2d = np.moveaxis(image_2d, 0, -1)

            image = wandb.Image(
                image_2d,
                masks={
                    "predictions": {
                        "mask_data": outputs_2d,
                    }
                },
            )
            image2 = wandb.Image(
                image_2d,
                masks={
                    "predictions": {
                        "mask_data": targets_2d,
                    }
                },
            )
            wandb.log({f"image{self.step_counter}": image, f"label{self.step_counter}": image2})

            if self.step_counter in [0, 99]:
                # 3D point cloud
                coords = output_dict["input"][:80000, :3].detach().cpu().numpy()
                outputs_3d = output_dict["outputs"][:80000].detach().cpu().numpy()
                # merge coordinates and labels
                pt_cloud = np.concatenate([coords, outputs_3d[:, None]], axis=1)
                # to wandb
                pt_cloud = wandb.Object3D({"type": "lidar/beta", "points": pt_cloud})
                wandb.log({f"pt_cloud{self.step_counter}": pt_cloud})

        self.step_counter += 1

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i], reduction="sum")
            self.total_correct[i] = dist.allreduce(self.total_correct[i], reduction="sum")
            self.total_positive[i] = dist.allreduce(self.total_positive[i], reduction="sum")

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (
                    self.total_seen[i] + self.total_positive[i] - self.total_correct[i]
                )
                ious.append(cur_iou)

        # WANDB
        wandb.log({"mean_iou": np.mean(ious) * 100})
        for i, iou_val in enumerate(ious):
            wandb.log({"iou_{}".format(i): iou_val * 100})

        miou = np.mean(ious)
        if hasattr(self, "trainer") and hasattr(self.trainer, "summary"):
            self.trainer.summary.add_scalar(self.name, miou * 100)
        else:
            print(ious)
            print(miou)

        # 2D mIoU calculation
        ciou, miou = compute_iou(self.val_mean_cm_student.compute())
        wandb.log({"mean_iou_2d": miou})
        for i, iou_val in enumerate(ciou):
            wandb.log({"iou_2d_{}".format(i): iou_val * 100})
