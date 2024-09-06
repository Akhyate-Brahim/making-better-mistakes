import torch
import torch.nn as nn
from typing import Dict, List

class SimpleHierarchyLoss(nn.Module):
    def __init__(self, lambda_coarse=1, lambda_fine=1, 
                 coarse_loss_start_epoch=0, coarse_loss_ramp_epochs=0,
                 fine_loss_start_epoch=0, fine_loss_ramp_epochs=0):
        super(SimpleHierarchyLoss, self).__init__()
        self.lambda_coarse = lambda_coarse
        self.lambda_fine = lambda_fine
        self.coarse_loss_start_epoch = coarse_loss_start_epoch
        self.coarse_loss_ramp_epochs = coarse_loss_ramp_epochs
        self.fine_loss_start_epoch = fine_loss_start_epoch
        self.fine_loss_ramp_epochs = fine_loss_ramp_epochs
        self.ce_loss = nn.CrossEntropyLoss()
        self.current_epoch = 0

    def forward(self, outputs, fine_labels):
        # Derive coarse labels
        coarse_labels = fine_labels // 5

        # Group fine predictions into coarse predictions
        coarse_outputs = outputs.view(-1, 20, 5).sum(dim=2)

        # Calculate coarse-grained loss weight
        coarse_weight = self._calculate_weight(self.coarse_loss_start_epoch, self.coarse_loss_ramp_epochs)

        # Calculate fine-grained loss weight
        fine_weight = self._calculate_weight(self.fine_loss_start_epoch, self.fine_loss_ramp_epochs)

        # Coarse-grained loss
        coarse_loss = self.ce_loss(coarse_outputs, coarse_labels)

        # Fine-grained loss
        fine_loss = self.ce_loss(outputs, fine_labels)

        # Combine losses
        total_loss = coarse_weight * self.lambda_coarse * coarse_loss + fine_weight * self.lambda_fine * fine_loss

        return total_loss

    def _calculate_weight(self, start_epoch, ramp_epochs):
        if self.current_epoch < start_epoch:
            weight = 0
        elif self.current_epoch < start_epoch + ramp_epochs:
            weight = (self.current_epoch - start_epoch) / ramp_epochs
        else:
            weight = 1
        return weight

    def update_epoch(self, epoch):
        self.current_epoch = epoch