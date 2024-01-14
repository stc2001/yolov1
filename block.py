import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A convolutional block used in the YOLOv1 architecture.

    This block consists of a convolutional layer followed by batch normalization
    and a Leaky ReLU activation function.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        batchnorm (nn.BatchNorm2d): Batch normalization layer.
        leakyrelu (nn.LeakyReLU): Leaky ReLU activation function.

    Parameters:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        """
        Defines the forward pass of the ConvBlock.

        Parameters:
            x (Tensor): Input tensor to the ConvBlock.

        Returns:
            Tensor: Output tensor after passing through the ConvBlock.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x

    


