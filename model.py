import torch
import torch.nn as nn
import sys
import logging
from block import ConvBlock



architecture = [
    # [from, number, module, args=(kernel_size, filter, stride, padding)]
    [-1, 1, ConvBlock, [7, 64, 2, 3]],
    [-1, 1, nn.MaxPool2d, [2, None, 2, None]],
    [-1, 1, ConvBlock, [3, 192, 1, 1]],
    [-1, 1, nn.MaxPool2d, [2, None, 2, None]],
    [-1, 1, ConvBlock, [1, 128, 1, 0]],
    [-1, 1, ConvBlock, [3, 256, 1, 1]],
    [-1, 1, ConvBlock, [1, 256, 1, 0]],
    [-1, 1, ConvBlock, [3, 512, 1, 1]],
    [-1, 1, nn.MaxPool2d, [2, None, 2, None]],
    [-1, 4, ConvBlock, [[1, 256, 1, 0], [3, 512, 1, 1]]],
    [-1, 1, ConvBlock, [1, 512, 1, 0]],
    [-1, 1, ConvBlock, [3, 1024, 1, 1]],
    [-1, 1, nn.MaxPool2d, [2, None, 2, None]],
    [-1, 2, ConvBlock, [[1, 512, 1, 0], [3, 1024, 1, 1]]],
    [-1, 1, ConvBlock, [3, 1024, 1, 1]],
    [-1, 1, ConvBlock, [3, 1024, 2, 1]],
    [-1, 1, ConvBlock, [3, 1024, 1, 1]],
    [-1, 1, ConvBlock, [3, 1024, 1, 1]],
]


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self.parse_module(self.architecture)
        self.fcs = self._create_fps(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def parse_module(self, architecture):
        layers = []
        in_channels = self.in_channels
        try:
            for layer in architecture:
                _, n, module, args = layer
                for i in range(n):
                    if module is ConvBlock:
                        if type(args[0])==list:
                            for arg in args:
                                kernel_size, out_channels, stride, padding = arg 
                                layers += [module(
                                    in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
                                    )]

                                in_channels = out_channels
                    
                        else:
                            kernel_size, out_channels, stride, padding = args 
                            layers += [module(
                                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
                                )]

                            in_channels = out_channels
                        
                    elif module is nn.MaxPool2d:
                        layers += [module(stride=2, kernel_size=2)]   
                    else:
                        logging.warning('Undefined Module: %s' %module) 

        except Exception as err:
            logging.error('Unexpected Error while parsing Conv layers: %s' %err)
        
        return nn.Sequential(*layers)

    def _create_fps(self, split_size, num_classes, num_boxes):
        S, C, B = split_size, num_classes, num_boxes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5))
        )

if __name__ == "__main__":
    yolov1 = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    x = torch.randn((2, 3, 448, 448))
    print(yolov1(x).shape)

