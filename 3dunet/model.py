import logging
import torch
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, ReLU, LeakyReLU, MaxPool3d
from torch import nn
from torchsummary import summary

logger = logging.getLogger(__name__)

class EncoderConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, concat_channels=0, first_level=False):
        super().__init__()
        self.conv1 = Conv3d(in_channels=in_channels+concat_channels, out_channels=out_channels//2, kernel_size=3, padding="same")
        self.bn1 = BatchNorm3d(num_features=out_channels//2)
        self.relu = ReLU()

        self.conv2 = Conv3d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, padding="same")
        self.bn2 = BatchNorm3d(num_features=out_channels)
        
        self.first_level = first_level
        if not first_level:
            self.pooling = MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input, concat=None):
        out = input
        if concat != None:
            out = torch.cat((out, concat), 1)
        
        if not self.first_level:
            out = self.pooling(out)

        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))

        return out

class DecoderConv3DBlock(nn.Module):
    def __init__(self, in_channels, concat_channels=0, first_level=False):
        super().__init__()
        self.Tconv1 = ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.conv1 = Conv3d(in_channels=in_channels+concat_channels, out_channels=in_channels//2, kernel_size=3, padding="same")
        self.bn = BatchNorm3d(num_features=in_channels//2)
        self.relu = ReLU()
        self.conv2 = Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding="same")
        
        self.first_level = first_level
        if first_level:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=1, kernel_size=1)

    def forward(self, input, concat=None):
        out = self.Tconv1(input)
        if concat != None:
            out = torch.cat((out, concat), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.first_level:
            out = self.conv3(out)
        return out
    
class TwoHeadedUNet3D(nn.Module):
    def __init__(self, filters=[64, 128, 256], bottleneck_filters=512):
        super().__init__()

        self.encoder0 = EncoderConv3DBlock(in_channels=1, out_channels=filters[0], first_level=True)
        
        self.head1_encoder1 = EncoderConv3DBlock(in_channels=filters[0], out_channels=filters[1])
        self.head1_encoder2 = EncoderConv3DBlock(in_channels=filters[1], out_channels=filters[2])

        self.head0_encoder1 = EncoderConv3DBlock(in_channels=filters[0], out_channels=filters[1], concat_channels=filters[0])
        self.head0_encoder2 = EncoderConv3DBlock(in_channels=filters[1], out_channels=filters[2], concat_channels=filters[1])

        self.bottleneck = EncoderConv3DBlock(in_channels=filters[2], out_channels=bottleneck_filters, concat_channels=filters[2])

        self.decoder2 = DecoderConv3DBlock(in_channels=bottleneck_filters, concat_channels=filters[2])
        self.decoder1 = DecoderConv3DBlock(in_channels=filters[2], concat_channels=filters[1])
        self.decoder0 = DecoderConv3DBlock(in_channels=filters[1], concat_channels=filters[0], first_level=True)

    def forward(self, input):
        head0_input = input[:, 0:1, :, :, :]
        head1_input = input[:, 1:2, :, :, :]

        head1_out0 = self.encoder0(head1_input)
        head1_out1 = self.head1_encoder1(head1_out0)
        head1_out2 = self.head1_encoder2(head1_out1)

        head0_out0 = self.encoder0(head0_input)
        head0_out1 = self.head0_encoder1(head0_out0, concat=head1_out0)
        head0_out2 = self.head0_encoder2(head0_out1, concat=head1_out1)

        out = self.bottleneck(head0_out2, concat=head1_out2)

        out = self.decoder2(out, concat=head0_out2)
        out = self.decoder1(out, concat=head0_out1)
        out = self.decoder0(out, concat=head0_out0)
        
        return out

class UNet3D(nn.Module):
    def __init__(self, in_channels, filters=[64, 128, 256], bottleneck_filters=512):
        super().__init__()
        
        self.encoder0 = EncoderConv3DBlock(in_channels=in_channels, out_channels=filters[0], first_level=True)
        self.encoder1 = EncoderConv3DBlock(in_channels=filters[0], out_channels=filters[1])
        self.encoder2 = EncoderConv3DBlock(in_channels=filters[1], out_channels=filters[2])

        self.bottleneck = EncoderConv3DBlock(in_channels=filters[2], out_channels=bottleneck_filters)

        self.decoder2 = DecoderConv3DBlock(in_channels=bottleneck_filters, concat_channels=filters[2])
        self.decoder1 = DecoderConv3DBlock(in_channels=filters[2], concat_channels=filters[1])
        self.decoder0 = DecoderConv3DBlock(in_channels=filters[1], concat_channels=filters[0], first_level=True)

    def forward(self, input):
        out0 = self.encoder0(input)
        out1 = self.encoder1(out0)
        out2 = self.encoder2(out1)

        out = self.bottleneck(out2)

        out = self.decoder2(out, concat=out2)
        out = self.decoder1(out, concat=out1)
        out = self.decoder0(out, concat=out0)
        return out

if __name__ == '__main__':
    model = TwoHeadedUNet3D(in_channels=2)
    summary(model=model.cuda(), input_size=(2, 96, 96, 96), batch_size=-1, device="cuda")
