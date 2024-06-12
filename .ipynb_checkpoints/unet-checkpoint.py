import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    # __init__ defines the layers and components of the network
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Downsampling Path
        self.down_convolution_1 = DownSample(in_channels, 64)  # First downsampling layer
        self.down_convolution_2 = DownSample(64, 128)          # Second downsampling layer
        self.down_convolution_3 = DownSample(128, 256)         # Third downsampling layer
        self.down_convolution_4 = DownSample(256, 512)         # Fourth downsampling layer

        # Bottleneck Layer
        self.bottle_neck = DoubleConv(512, 1024)

        # Upsampling Path
        self.up_convolution_1 = UpSample(1024, 512)            # First upsampling layer
        self.up_convolution_2 = UpSample(512, 256)             # Second upsampling layer
        self.up_convolution_3 = UpSample(256, 128)             # Third upsampling layer
        self.up_convolution_4 = UpSample(128, 64)              # Fourth upsampling layer

        # Output Layer
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    # forward defines how the input tensor flows through the layers
    def forward(self, x):
       # Downsampling Path
       down_1, p1 = self.down_convolution_1(x)                  # First downsampling layer
       down_2, p2 = self.down_convolution_2(p1)                 # Second downsampling layer
       down_3, p3 = self.down_convolution_3(p2)                 # Third downsampling layer
       down_4, p4 = self.down_convolution_4(p3)                 # Fourth downsampling layer

       # Bottleneck
       b = self.bottle_neck(p4)

       # Upsampling Path
       up_1 = self.up_convolution_1(b, down_4)                  # First upsampling layer
       up_2 = self.up_convolution_2(up_1, down_3)               # Second upsampling layer
       up_3 = self.up_convolution_3(up_2, down_2)               # Third upsampling layer
       up_4 = self.up_convolution_4(up_3, down_1)               # Fourth upsampling layer

       # Output Layer
       out = self.out(up_4)
       return out

"""
if __name__ == "__main__":
    double_conv = DoubleConv(256, 256)
    print(double_conv)

    input_image = torch.rand((1,3, 512, 512))
    model = UNet(3, 10)
    output = model(input_image)
    print(output.size()) # Expected size: [1, 10, 512, 512]
"""