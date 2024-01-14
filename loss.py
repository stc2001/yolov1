import torch
import torch.nn as nn
from metrics import iou

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    """
    The YOLOv1 loss function as described in the original paper.
    It computes the loss for classification, localization, and confidence.
    """

    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        """
        Initialization of YOLOv1 loss function parameters.

        Parameters:
            S (int): Number of grid cells along width and height (SxS grid).
            B (int): Number of bounding boxes per grid cell.
            C (int): Number of classes.
            lambda_coord (float): Weight for localization loss.
            lambda_noobj (float): Weight for confidence loss when no object is present in the cell.
        """
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        """
        Computes the YOLOv1 loss function.

        Parameters:
            predictions (torch.Tensor): Predictions from the model (batch_size, S*S*(C+B*5)).
            target (torch.Tensor): Ground truth (batch_size, S*S*(C+B*5)).

        Returns:
            torch.Tensor: Total loss for the batch.
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_box1 = iou(target[..., self.C+1:self.C+5], predictions[..., self.C+1:self.C+5])
        iou_box2 = iou(target[..., self.C+1:self.C+5], predictions[..., self.C+6:self.C+10]) 
        ious = torch.cat((iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)), dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0) 
        exists_box = target[..., self.C].unsqueeze(3) # identity of object i (if there is an object in cell i)


        ###################
        # Box Coordinates #
        ###################
        box_predictions = exists_box * (
            (
                best_box * predictions[..., self.C+6:self.C+10] + 
                (1 - best_box) * predictions[..., self.C+1:self.C+5]
            )
        )

        box_targets = exists_box * target[..., self.C+1:self.C+5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        ) 

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            #  (N, S, S, 4) -> (N * S * S, 4)
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        ###############
        # object loss #
        ###############
        pred_box = (
            best_box * predictions[..., self.C+5:self.C+6] + 
            (1 - best_box) * predictions[..., self.C:self.C+1]
        )

        obj_loss = self.mse(
            # (N, S, S, 1) -> (N * S * S, 1)
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C+1])
        )

        ##################
        # No Object Loss #
        ##################
        no_object_loss = self.mse(
            # (N, S, S, 1) -> (N, S*S)
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        no_object_loss += self.mse(
            # (N, S, S, 1) -> (N, S*S)
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)   
        )
        

        ##############
        # Class Loss #
        ##############
        class_loss = self.mse(
            # (N, S, S, 20) -> (N*S*S, 20)
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        ##############
        # Total Loss #
        ##############
        loss = (
            self.lambda_coord * box_loss + 
            obj_loss + 
            self.lambda_coord * no_object_loss + 
            class_loss
        )

        return loss
