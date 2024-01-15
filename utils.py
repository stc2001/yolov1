import torch
from metrics import iou, nms

def convert_predboxes(predictions, S=7, B=2, C=20):
    """
    Convert YOLO model output to original label format.

        Args:
        predictions (torch.Tensor): The output tensor from a YOLO model of shape (N, S, S, C + B * 5),
                                    where S is the grid size, and C is the number of classes.
                                    The last 5 elements in the tensor are the objectness score,
                                    and the normalized bounding box coordinates [x, y, w, h].
        S (int): The size of the grid (number of cells in one dimension).
        C (int): The number of classes.
        B (int): the number of boxes per cell

        Returns:
        list: A list of detections, where each detection is represented as a list:
            [class_id, conf, x_center, y_center, width, height].
            The coordinates are normalized with respect to the image size.
    """

    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape((batch_size, S, S, C + B * 5))

    box1 = predictions[..., C+1:C + 5]
    box2 = predictions[..., C+6:C + B * 5]
    scores = torch.cat(
        (
            predictions[..., C].unsqueeze(0), 
            predictions[..., C+5].unsqueeze(0)
        ), 
        dim=0
    )

    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = box1 * (1 - best_box) + box2 * best_box

    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h =  1 / S * best_boxes[..., 2:]
    converted_boxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C+1], predictions[..., C+5]).unsqueeze(-1)
    converted_boxes =  torch.cat((predicted_class, best_confidence, converted_boxes), dim=-1)

    return converted_boxes


def convert_trueboxes(true_boxes, S=7, C=20):
    """
    Convert YOLO model label to original label format.

        Args:
        true_boxes (torch.Tensor): true boxes (N, S, S, C + 5),
                                    where S is the grid size, and C is the number of classes.
                                    The last 5 elements in the tensor are the objectness score,
                                    and the normalized bounding box coordinates [x, y, w, h].
        S (int): The size of the grid (number of cells in one dimension).
        C (int): The number of classes.

        Returns:
        list: A list of true boxes, where each box is represented as a list:
            [class_id, x_center, y_center, width, height].
            The coordinates are normalized with respect to the image size.
    """
    batch_size = true_boxes.shape[0]
    box = true_boxes[..., C+1:C+5]

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = (1 / S) * (box[..., :1] + cell_indices)
    y = (1 / S) * (box[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = (1 / S) * box[..., 2:]
    converted_boxes = torch.cat((x, y, w_h), dim=-1)
    class_id = true_boxes[..., :C].argmax(-1).unsqueeze(-1)
    objectness = true_boxes[..., C:C+1]
    converted_boxes = torch.cat((class_id, objectness, converted_boxes), dim=-1)

    return converted_boxes

def cellboxes_to_boxes(cellboxes, S=7):
    # (N, S, S, 6) --> (N, S*S, 6)
    cellboxes = cellboxes.reshape(cellboxes.shape[0], S*S, -1)

    all_boxes = []
    for batch_idx in range(cellboxes.shape[0]):
        boxes = []
        for bbx_idx in range(S*S):
            boxes.append([x.item() for x in cellboxes[batch_idx, bbx_idx, :]])
        
        all_boxes.append(boxes)
    
    return all_boxes

    

def get_boxes(dataloader, model, S=7, C=20, device='cpu'):
    """
    Convert YOLO model label to original label format.

        Args:
        dataloader (torch.Tensor): torch Dataloader (N, S, S, C + 5),
                                    where S is the grid size, and C is the number of classes.
                                    The last 5 elements in the tensor are the objectness score,
                                    and the normalized bounding box coordinates [x, y, w, h].
        S (int): The size of the grid (number of cells in one dimension).
        C (int): The number of classes.

        Returns:
        list: A list of true boxes, where each box is represented as a list:
            [class_id, x_center, y_center, width, height].
            The coordinates are normalized with respect to the image size.
    """
    all_true_boxes = []
    all_pred_boxes = []

    train_idx = 0
    model.eval()
    for batch_idx, (x, labels) in enumerate(dataloader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(x)

        batch_size = x.shape[0]
        true_boxes = cellboxes_to_boxes(
            convert_trueboxes(true_boxes, S=S, C=C),
            S=S
        )

        boxes = cellboxes_to_boxes(
            convert_cellboxes(out, S=S, C=C, B=2), 
            S=S,
        )

        for idx in range(batch_idx):
            nms_boxes = nms(
                boxes[idx], iou_threshold=0.5, conf=0.4, xywh=True
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            
            for box in true_boxes[idx]:
                all_true_boxes.append([train_idx] + box)
            
            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes    



if  __name__ == "__main__":
    predictions = torch.randn((8, 7, 7, 25))
    converted_boxes = convert_trueboxes(predictions, S=7, C=20)

    all_boxes = cellboxes_to_boxes(converted_boxes)

    print(all_boxes[0][0])



    