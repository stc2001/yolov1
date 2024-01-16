import os
import cv2
import torch
import numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
from dataset import Dataset, Compose
from utils import convert_trueboxes, cellboxes_to_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()



S = 33
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
train_data = Dataset(
    path='../DATA',
    mode='train',
    C=12,
    S=S,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=False)
# train_features, train_labels = next(iter(trainloader))
# print(f"Feature Batch Shape: {train_features.shape}")
# print(f"Label Batch Shape: {train_labels.shape}")


all_boxes = []
train_idx = 0
for batch_idx, (x, labels, num_boxes) in enumerate(trainloader):
    converted_boxes = convert_trueboxes(
        labels, S=S, C=12
    )

    true_boxes = cellboxes_to_boxes(
        converted_boxes, S=S
    )

    batch_size = x.shape[0]
    for idx in range(batch_size):
        boxes = []
        for box in true_boxes[idx]:
            if box[1] > 0.5:
                boxes.append(box)
            all_boxes += boxes

        print(f"{num_boxes[idx]} - {len(boxes)}")
        plot_image(x[idx].permute(1,2,0).to("cpu"), boxes)
        train_idx += 0

