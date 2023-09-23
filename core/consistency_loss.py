import torch
import torch.nn as nn


class PartialConsistencyLoss(nn.Module):
    def __init__(self, H, ignore_index=0, beta=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.supervised_loss = H(ignore_index=ignore_index, reduction="sum")
        self.consistency_loss = nn.KLDivLoss(reduction="sum")
        self.beta = 0.4

    def forward(self, student_output, teacher_output, student_label):
        loss_s = self.compute_supervised_loss(student_output, student_label)
        mask = student_label == self.ignore_index
        loss_u = self.compute_consistency_loss(student_output, teacher_output, mask=mask)
        return ((1 - self.beta) * loss_s + self.beta * loss_u) / student_output.shape[0]
        # return loss_s / student_output.shape[0]

    def compute_supervised_loss(self, student_output, student_label):
        return self.supervised_loss(student_output, student_label)

    def compute_consistency_loss(self, student_output, teacher_output, mask=None):
        student_output_reshaped = student_output.log_softmax(-1)
        teacher_output_reshaped = teacher_output.softmax(-1)
        return self.consistency_loss(student_output_reshaped[mask], teacher_output_reshaped[mask])


class PartialConsistencyLoss2D(nn.Module):
    def __init__(self, H, ignore_index=0, beta=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.supervised_loss = H(ignore_index=ignore_index, reduction="sum")
        self.consistency_loss = nn.KLDivLoss(reduction="sum")
        self.beta = beta

    def forward(self, student_output, teacher_output, student_label, teacher_weight):
        loss_s = self.compute_supervised_loss(student_output, student_label)
        mask = student_label == self.ignore_index

        loss_u = self.compute_consistency_loss(student_output, teacher_output, mask=mask)
        return ((1 - self.beta) * loss_s + self.beta * loss_u) / student_output.shape[-1]
        # return loss_s / student_output.shape[-1]

    def compute_supervised_loss(self, student_output, student_label):
        return self.supervised_loss(student_output, student_label)

    def compute_consistency_loss(self, student_output, teacher_output, mask=None):
        student_output_reshaped = student_output.permute(0, 2, 3, 1).log_softmax(-1)
        teacher_output_reshaped = teacher_output.permute(0, 2, 3, 1).softmax(-1)
        return self.consistency_loss(student_output_reshaped[mask], teacher_output_reshaped[mask])
