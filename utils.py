import torch



def convert_cellboxes(predictions, S=7, B=2, C=20):
    """
    Convert YOLO model output to original label format.

        Args:
        predictions (torch.Tensor): The output tensor from a YOLO model of shape (N, S, S, C + B * 5),
                                    where S is the grid size, and C is the number of classes.
                                    The last 5 elements in the tensor are the objectness score,
                                    and the normalized bounding box coordinates [x, y, w, h].
        S (int): The size of the grid (number of cells in one dimension).
        C (int): The number of classes.

        Returns:
        list: A list of detections, where each detection is represented as a list:
            [class_id, x_center, y_center, width, height].
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
    print(best_box.shape)
    print(box1.shape)
    print(box2.shape)
    best_boxes = box1 * (1 - best_box) + box2 * best_box

    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] * cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] * cell_indices)
    w_h =  1 / S * best_boxes[..., 2:]
    converted_boxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C+1], predictions[..., C+5]).unsqueeze(-1)
    converted_boxes =  torch.cat((predicted_class, best_confidence, converted_boxes), dim=-1)

    return converted_boxes



if  __name__ == "__main__":
    predictions = torch.randn((8, 7, 7, 30))
    converted_boxes = convert_cellboxes(predictions, S=7, C=20,  B=2)

    print(converted_boxes.shape)



    