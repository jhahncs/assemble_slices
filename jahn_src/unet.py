"""Lightning module for a 3D u-net"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def create_double_conv(channels):
    """Create a double convolution+BN+ReLU block

    Parameters
    ----------
    channels : list or tuple
        Holds channels of input, middle and output layer of encoder block

    Returns
    -------
    DoubleConv : nn.Sequential
    """
    print('create_double_conv',channels[0], channels[1], 3)
    return nn.Sequential(
        nn.Conv3d(channels[0], channels[1], 3),
        nn.BatchNorm3d(channels[1]),
        nn.ReLU(),
        nn.Conv3d(channels[1], channels[2], 3),
        nn.BatchNorm3d(channels[2]),
        nn.ReLU(),
    )


class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define encoder convolution blocks
        self.encoder_conv1 = create_double_conv([in_channels, 32, 64])
        self.encoder_conv2 = create_double_conv([64, 64, 128])
        self.encoder_conv3 = create_double_conv([128, 128, 256])
        self.encoder_conv4 = create_double_conv([256, 256, 512])

        # Define decoder convolution blocks
        self.decoder_conv1 = create_double_conv([256 + 512, 256, 256])
        self.decoder_conv2 = create_double_conv([128 + 256, 128, 128])
        self.decoder_conv3 = create_double_conv([64 + 128, 64, 64])

        # Define maxpool
        self.maxpool = nn.MaxPool3d(2)  # Kernel size and stride 2

        # Define upconv
        self.upconv1 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        # Define final convolution
        self.outconv = nn.Conv3d(64, self.out_channels, 1)

        # Define loss criterion
        self.criterion = nn.BCEWithLogitsLoss()

    def center_crop(self, x_encoder, x):
        """Center crop a tensor

        Parameters
        ----------
        x_encoder : tensor
            The tensor at the end of each encoder convblock that is 
            cropped.
        x : tensor
            The tensor at the start of each decoder convblock that is
            concatenated with the cropped x_encoder.

        Returns
        -------
        out : tensor
            Cropped x_encoder
        """
        crop1 = (x_encoder.shape[2] - x.shape[2]) // 2  # First dimension
        crop2 = (x_encoder.shape[3] - x.shape[3]) // 2  # Second dimension
        crop3 = (x_encoder.shape[4] - x.shape[4]) // 2  # Third dimension

        return F.pad(x_encoder, (-crop3, -crop3, -crop2, -crop2, -crop1, -crop1))

    def forward(self, x):
        # Encoder
        print("1",x.shape)
        x1 = self.encoder_conv1(x)
        print("2",x1.shape)
        x = self.maxpool(x1)
        print("3",x.shape)
        x2 = self.encoder_conv2(x)
        print("3",x2.shape)
        x = self.maxpool(x2)
        print('4',x.shape)
        x3 = self.encoder_conv3(x)
        x = self.maxpool(x3)
        x = self.encoder_conv4(x)

        # Decoder
        x = self.upconv1(x)
        x = torch.cat((self.center_crop(x3, x), x), dim=1)
        x = self.decoder_conv1(x)
        x = self.upconv2(x)
        x = torch.cat((self.center_crop(x2, x), x), dim=1)
        x = self.decoder_conv2(x)
        x = self.upconv3(x)
        x = torch.cat((self.center_crop(x1, x), x), dim=1)
        x = self.decoder_conv3(x)

        # Final convolution
        x = self.outconv(x)

        return x
