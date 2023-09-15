from typing import Any, Callable, Dict
import copy

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

from core.lovasz import lovasz_softmax, lovasz_softmax_flat

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
        self.initialize_teacher()
        self.criterion = criterion
        self.criterion_lovasz = lovasz_softmax_flat
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1

    def _before_epoch(self) -> None:
        self.student.train()
        self.teacher.eval()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id
        )

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if "name" not in key:
                _inputs[key] = value.cuda()

        inputs_student = _inputs["lidar"]
        inputs_teacher = _inputs["lidar_teacher"]
        targets = feed_dict["targets"].F.long().cuda(non_blocking=True)

        self.update_teacher()

        with amp.autocast(enabled=self.amp_enabled):
            outputs_student = self.student(inputs_student)
            outputs_teacher = self.teacher(inputs_teacher)

            if outputs_student.requires_grad:
                # Modify teacher output
                # inv_map_teacher = feed_dict["inverse_map_teacher"].F
                # outputs_teacher = outputs_teacher[inv_map_teacher]
                # forw_map_student = feed_dict["forward_map"].F
                # out_teacher_mapped_student = outputs_teacher[forw_map_student]

                # t_labels = _inputs["targets_teacher"].F[inv_map_teacher]
                # t_labels_app_student = t_labels[forw_map_student]

                out_teacher_mapped_student = torch.zeros_like(outputs_student)
                t_labels_app_student = torch.zeros_like(targets)
                for idx in range(feed_dict["forward_map"].C[:, 0].max() + 1):
                    inv_map_batch_idx = feed_dict["inverse_map_teacher"].C[:, 0] == idx
                    out_batch_idx = feed_dict["forward_map_teacher"].C[:, 0] == idx
                    stud_batch_idx = feed_dict["forward_map"].C[:, 0] == idx

                    inv_map_teacher_batch = feed_dict["inverse_map_teacher"].F[inv_map_batch_idx]
                    outputs_teacher_batch = outputs_teacher[out_batch_idx][inv_map_teacher_batch]

                    forw_map_student_batch = feed_dict["forward_map"].F[stud_batch_idx]
                    out_teacher_mapped_student_batch = outputs_teacher_batch[forw_map_student_batch]
                    out_teacher_mapped_student[stud_batch_idx] = out_teacher_mapped_student_batch

                    t_labels_batch = _inputs["targets_teacher"].F[out_batch_idx][
                        inv_map_teacher_batch
                    ]  #
                    t_labels_app_student_batch = t_labels_batch[forw_map_student_batch]
                    t_labels_app_student[stud_batch_idx] = t_labels_app_student_batch

                # Remove invalid points
                valid = targets != 255
                voutputs_student = outputs_student[valid.nonzero().squeeze()]
                vtargets = targets[valid]

                loss = self.criterion(
                    outputs_student, out_teacher_mapped_student, targets
                ) + self.criterion_lovasz(voutputs_student, vtargets)

        if outputs_student.requires_grad:
            self.summary.add_scalar("loss", loss.item())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        else:
            invs = feed_dict["inverse_map"]
            all_labels = feed_dict["targets_mapped"]
            _outputs = []
            _targets = []
            for idx in range(invs.C[:, 0].max() + 1):
                cur_scene_pts = (inputs_student.C[:, 0] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, 0] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, 0] == idx).cpu().numpy()
                outputs_mapped = outputs_student[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels.F[cur_label]
                _outputs.append(outputs_mapped)
                _targets.append(targets_mapped)
            outputs_student = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)

        return {"outputs": outputs_student, "targets": targets, "outputs_teacher": outputs_teacher}

    def _after_epoch(self) -> None:
        self.student.eval()

    def initialize_teacher(self) -> None:
        self.alpha = 0.999  # TODO: Move to config
        for p in self.teacher.parameters():
            p.detach_()

    def update_teacher(self) -> None:
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
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
