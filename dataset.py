import os
import cv2
import torch
import numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train', S=7, B=2, C=20, transform=None):
        self.path = path
        self.mode = mode
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.images = sorted(os.listdir(self.path + "/" + self.mode + "/images"))
        self.labels = sorted(os.listdir(self.path + "/" + self.mode + "/labels"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = os.path.join(self.path, self.mode, 'labels', self.labels[index])
        boxes = self.check_txtfile(label)

        image = os.path.join(self.path, self.mode, 'images', self.images[index])
        image = Image.open(image)
        boxes = torch.tensor(boxes)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)


        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            class_id, x, y, w, h = box.tolist()
            class_id = int(class_id)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = (
                w * self.S,
                h * self.S
            )
            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                label_matrix[i, j, self.C+1:self.C+5] =  torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )
                label_matrix[i, j, class_id] = 1
                
        return  image, label_matrix

    def check_txtfile(self, txtfile):
        # check if txtfile exist
        boxes = []
        if not os.path.exists(txtfile):
            return boxes

        with open(txtfile, 'r') as f:
            for label in f.readlines():
                class_id, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_id, x, y, w, h])
        
        return boxes


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes):
        for transform in self.transforms:
            image, bboxes = transform(image), bboxes
        return image, bboxes

if __name__ == "__main__":
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    train_data = Dataset(
        path='../DATA',
        mode='train',
        C=12,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    train_features, train_labels = next(iter(trainloader))
    print(f"Feature Batch Shape: {train_features.shape}")
    print(f"Label Batch Shape: {train_labels.shape}")


    import matplotlib.pyplot as plt
    one_image = train_features[0]
    one_image = one_image.permute(1, 2, 0).to('cpu')
    plt.imshow(one_image)
    plt.show()
