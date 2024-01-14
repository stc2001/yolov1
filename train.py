import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from loss import Loss
from dataset import Dataset, Compose
from model import Yolov1
from metrics import iou, nms, mean_average_precision


LEARNING_RATE = 1e-4
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_OF_WORKERS = 2
PIN_MEMORY = True
PATH = '../DATA'


def train_fn(model, train_loader, val_loader, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")



def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=12).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = Loss(C=12)

    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    train_dataset = Dataset(
        path=PATH,
        mode='train',
        C=12,
        transform=transform
    )

    test_dataset = Dataset(
        path=PATH,
        mode='valid',
        C=12,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_OF_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_OF_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        train_fn(model, train_loader, test_loader, optimizer, loss_fn)


if __name__ == "__main__":
    main()
