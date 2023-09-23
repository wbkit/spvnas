from typing import Any, Callable, Dict
import copy

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from transformers import SegformerForSemanticSegmentation

from core.consistency_loss import PartialConsistencyLoss2D

# from core.lovasz import lovasz_softmax, lovasz_softmax_flat

__all__ = ["SemanticKITTITrainer"]


class SemanticKITTITrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: Optimizer,
        scheduler: Scheduler,
        num_workers: int,
        seed: int,
        amp_enabled: bool = False,
    ) -> None:
        self.student = model
        self.teacher = copy.deepcopy(self.student)
        ## Add 2D part
        self.teacher_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            return_dict=False,
            num_labels=16,
            ignore_mismatched_sizes=True,
        ).cuda()
        self.student_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            return_dict=False,
            num_labels=16,
            ignore_mismatched_sizes=True,
        ).cuda()
        self.initialize_teacher()
        self.criterion = criterion
        # self.criterion_lovasz = lovasz_softmax_flat
        self.criterion2D = PartialConsistencyLoss2D(nn.CrossEntropyLoss, ignore_index=255, beta=0.1)
        self.optimizer = optimizer
        self.optimizer2D = torch.optim.Adam(
            [p for p in self.student_2d.parameters() if p.requires_grad], lr=2e-05, eps=1e-08
        )
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.scaler_2d = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

    def _before_epoch(self) -> None:
        self.student.train()
        self.teacher.eval()
        self.student_2d.train()
        # self.teacher_2d.eval()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id
        )

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if "name" not in key:
                _inputs[key] = value.cuda()

        inputs_student = _inputs["student_lidar"]
        inputs_teacher = _inputs["teacher_lidar"]
        targets = feed_dict["teacher_targets"].F.long().cuda(non_blocking=True)

        inputs_2d_student = _inputs["student_pixel_values"]
        inputs_2d_teacher = _inputs["teacher_pixel_values"]
        targets_2d = _inputs["teacher_labels"]

        self.update_teacher()

        # if logits_student_2d.requires_grad:
        #     self.optimizer.zero_grad()
        #     loss_2d.backward()
        #     self.optimizer2D.step()

        # 3D
        with amp.autocast(enabled=self.amp_enabled):
            outputs_student = self.student(inputs_student)
            outputs_teacher = self.teacher(inputs_teacher)

            (logits_student_2d,) = self.student_2d(inputs_2d_student)
            (logits_teacher_2d,) = self.teacher_2d(inputs_2d_teacher)
            upsampled_logits_student = nn.functional.interpolate(
                logits_student_2d,
                size=targets_2d.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            upsampled_logits_teacher = nn.functional.interpolate(
                logits_teacher_2d,
                size=targets_2d.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            if outputs_student.requires_grad:
                # Modify teacher output
                # inv_map_teacher = feed_dict["teacher_inverse_map"].F
                # outputs_teacher = outputs_teacher[inv_map_teacher]
                # forw_map_student = feed_dict["student_forward_map"].F
                # out_teacher_mapped_student = outputs_teacher[forw_map_student]

                # t_labels = _inputs["teacher_targets"].F[inv_map_teacher]
                # t_labels_app_student = t_labels[forw_map_student]

                out_teacher_mapped_student = torch.zeros_like(outputs_student)
                t_labels_app_student = torch.zeros_like(targets)
                for idx in range(feed_dict["student_forward_map"].C[:, 0].max() + 1):
                    inv_map_batch_idx = feed_dict["teacher_inverse_map"].C[:, 0] == idx
                    out_batch_idx = feed_dict["teacher_forward_map"].C[:, 0] == idx
                    stud_batch_idx = feed_dict["student_forward_map"].C[:, 0] == idx

                    inv_map_teacher_batch = feed_dict["teacher_inverse_map"].F[inv_map_batch_idx]
                    outputs_teacher_batch = outputs_teacher[out_batch_idx][inv_map_teacher_batch]

                    forw_map_student_batch = feed_dict["student_forward_map"].F[stud_batch_idx]
                    out_teacher_mapped_student_batch = outputs_teacher_batch[forw_map_student_batch]
                    out_teacher_mapped_student[stud_batch_idx] = out_teacher_mapped_student_batch

                    t_labels_batch = _inputs["teacher_targets"].F[out_batch_idx][
                        inv_map_teacher_batch
                    ]  #
                    t_labels_app_student_batch = t_labels_batch[forw_map_student_batch]
                    t_labels_app_student[stud_batch_idx] = t_labels_app_student_batch

                # Remove invalid points
                # valid = targets != 255
                # voutputs_student = outputs_student[valid.nonzero().squeeze()]
                # vtargets = targets[valid]

                loss = self.criterion(
                    outputs_student, out_teacher_mapped_student, targets
                )  # + 0.1 * self.criterion_lovasz(voutputs_student, vtargets)

                ## 2D
                loss_2d = self.criterion2D(
                    upsampled_logits_student, upsampled_logits_teacher, targets_2d, 1
                )

        if outputs_student.requires_grad:
            self.summary.add_scalar("loss", loss.item())
            self.summary.add_scalar("loss_2d", loss_2d.item())

            self.optimizer.zero_grad()
            self.optimizer2D.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler_2d.scale(loss_2d).backward()
            self.scaler.step(self.optimizer)
            self.scaler_2d.step(self.optimizer2D)
            self.scaler.update()
            self.scaler_2d.update()
            self.scheduler.step()

            # self.optimizer2D.zero_grad()
            # loss_2d.backward()
            # self.optimizer2D.step()

        else:
            invs = feed_dict["student_inverse_map"]
            all_labels = feed_dict["student_targets_mapped"]
            _outputs = []
            _targets = []
            _inputs = []
            for idx in range(invs.C[:, 0].max() + 1):
                cur_scene_pts = (inputs_student.C[:, 0] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, 0] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, 0] == idx).cpu().numpy()
                outputs_mapped = outputs_student[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                inputs_mapped = inputs_student.F[cur_scene_pts][cur_inv]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
                _inputs.append(inputs_mapped)
            outputs_student = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)
            inputs_student = torch.cat(_inputs, 0)

        return {
            "outputs": outputs_student,
            "targets": targets,
            "outputs_teacher": outputs_teacher,
            "input": inputs_student,
            "outputs_2d": upsampled_logits_student,
            "targets_2d": targets_2d,
            "images": inputs_2d_student,
        }

    def _after_epoch(self) -> None:
        self.student.eval()
        self.student_2d.eval()

    def initialize_teacher(self) -> None:
        self.alpha = 0.99  # TODO: Move to config
        for p in self.teacher.parameters():
            p.detach_()

        for p in self.teacher_2d.parameters():
            p.detach_()

    def update_teacher(self) -> None:
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        # 3D
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(alpha).add_(1 - alpha, sp.data)
        # 2D
        for tp, sp in zip(self.teacher_2d.parameters(), self.student_2d.parameters()):
            tp.data.mul_(alpha).add_(1 - alpha, sp.data)

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["model"] = self.student.state_dict()
        state_dict["scaler"] = self.scaler.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["scheduler"] = self.scheduler.state_dict()
        state_dict["model_teacher"] = self.teacher.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.student.load_state_dict(state_dict["model"])
        self.scaler.load_state_dict(state_dict.pop("scaler"))
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.teacher.load_state_dict(state_dict["model_teacher"])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass
