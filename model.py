import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, out_channels):
        super().__init__()
        self.g_w = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.x_w = nn.Sequential(
            nn.Conv2d(x_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.g_w(g)
        x1 = self.x_w(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MobileNetV2(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, pretrained=True):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)

        # feature pyramid
        # skip-connection
        self.enc1 = nn.Sequential(*list(mobilenet.features.children())[:2]) # 128x128, 16ch
        self.enc2 = nn.Sequential(*list(mobilenet.features.children())[2:4]) # 64x64, 24ch
        self.enc3 = nn.Sequential(*list(mobilenet.features.children())[4:7]) # 32x32, 32ch
        self.enc4 = nn.Sequential(*list(mobilenet.features.children())[7:14]) # 16x16, 96ch
        self.bottleneck = nn.Sequential(*list(mobilenet.features.children())[14:18]) # 8x8, 320 ch

        # адаптация каналов для декодера
        self.adapt1 = nn.Conv2d(16, 64, 1) #enc1
        self.adapt2 = nn.Conv2d(24, 128, 1) # enc2
        self.adapt3 = nn.Conv2d(32, 256, 1) # enc3
        self.adapt4 = nn.Conv2d(96, 512, 1) # enc4
        self.adapt_b = nn.Conv2d(320, 1024, 1) # bottleneck

        # decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionBlock(512, 512, 512)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionBlock(256, 256, 256)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionBlock(128, 128, 128)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionBlock(64, 64, 64)
        self.dec1 = ConvBlock(128, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = ConvBlock(35, 32)

        self.output = nn.Conv2d(32, out_channel, 1)

        if pretrained:
            self.freeze_encoder()

    def freeze_encoder(self):
        '''
        freeze layers (trainable = False)
        '''
        for param in self.enc1.parameters():
            param.requires_grad = False
        for param in self.enc2.parameters():
            param.requires_grad = False
        for param in self.enc3.parameters():
            param.requires_grad = False
        for param in self.enc4.parameters():
            param.requires_grad = False
        for param in self.bottleneck.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        '''
        unfreeze layers (trainable = True) for fine-tuning
        '''
        for param in self.enc1.parameters():
            param.requires_grad = True
        for param in self.enc2.parameters():
            param.requires_grad = True
        for param in self.enc3.parameters():
            param.requires_grad = True
        for param in self.enc4.parameters():
            param.requires_grad = True
        for param in self.bottleneck.parameters():
            param.requires_grad = True

    def forward(self, x):
        '''
        Train encoder (mobileNetV2)
        :param x:
        :return:
        '''
        e1_raw = self.enc1(x)
        e2_raw = self.enc2(e1_raw)
        e3_raw = self.enc3(e2_raw)
        e4_raw = self.enc4(e3_raw)
        b_raw = self.bottleneck(e4_raw)

        # adaptive
        e0 = x
        e1 = self.adapt1(e1_raw)
        e2 = self.adapt2(e2_raw)
        e3 = self.adapt3(e3_raw)
        e4 = self.adapt4(e4_raw)
        b = self.adapt_b(b_raw)

        # Decoder + attention
        d4 = self.up4(b)
        d4 = self.att4(d4, e4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.att3(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.att2(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.att1(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        return torch.sigmoid(self.output(d0))

if __name__ == '__main__':
    print('Test MobileNetV2')
    model = MobileNetV2(pretrained=True)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print(f'Input: {x.shape} - Output: {y.shape}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    frozen = sum(1 for p in model.enc1.parameters() if not p.requires_grad)
    print(f'Frozen params in enc1: {frozen}')