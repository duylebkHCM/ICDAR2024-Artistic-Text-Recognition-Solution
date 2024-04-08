import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_MLA(nn.Module):
    def __init__(self, in_channels=1024, mla_channels=256):
        super(Conv_MLA, self).__init__()
        self.mla_p2_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4_1x1 = nn.Sequential(nn.Conv2d(
            in_channels, mla_channels, 1, bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

        self.mla_p2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p3 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())
        self.mla_p4 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1,
                                              bias=False), nn.BatchNorm2d(mla_channels), nn.ReLU())

    def forward(self, res2, res3, res4):
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p3_plus = mla_p4_1x1 + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p4 = self.mla_p4(mla_p4_1x1)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return mla_p2, mla_p3, mla_p4

class MLAHead(nn.Module):
    def __init__(self, in_channels=384, mla_channels=128, mlahead_channels=64):
        super(MLAHead, self).__init__()

        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
          )
        self.head3 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
            )
        self.head4 = nn.Sequential(
            nn.Conv2d(in_channels, mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
            )
        
    def forward(self, mla_p2, mla_p3, mla_p4):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        return torch.cat([head2, head3, head4], dim=1)


class MLAHeadv2(nn.Module):
    def __init__(self, in_channels=[256, 384, 384], mla_channels=128, mlahead_channels=64):
        super(MLAHead, self).__init__()

        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels[0], mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
          )
        self.head3 = nn.Sequential(
            nn.Conv2d(in_channels[1], mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
            )
        self.head4 = nn.Sequential(
            nn.Conv2d(in_channels[2], mla_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mla_channels),
            nn.ReLU(),
            nn.Conv2d(mla_channels, mlahead_channels, 1, bias=False),
            nn.BatchNorm2d(mlahead_channels),
            nn.ReLU()
            )
        
    def forward(self, mla_p2, mla_p3, mla_p4):
        head2 = self.head2(mla_p2)
        head3 = self.head3(mla_p3)
        head4 = self.head4(mla_p4)
        return head2, head3, head4
    
    
class SegHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, in_channels=384, mla_channels=128, mlahead_channels=64, num_classes=2, **kwargs):
        super(SegHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.conv_mla = Conv_MLA(in_channels, mla_channels)
        self.mlahead = MLAHead(in_channels=in_channels, mla_channels=mla_channels, mlahead_channels=mlahead_channels)
        self.unpool1 = nn.Sequential(nn.ConvTranspose2d(192, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.unpool2 = nn.Sequential(nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.cls = nn.Conv2d(128, self.num_classes, 3, padding=1)

    def forward(self, inputs):
        x = self.mlahead(inputs[0], inputs[1], inputs[2])
        x = self.unpool1(x)
        x = self.unpool2(x)
        x = self.cls(x)
        return x


class SegHeadV2(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, in_channels=512, num_classes=2, **kwargs):
        super(SegHeadV2, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        self.upconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels, 256, (4, 1), (2, 1), (1, 0)),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))
        self.doubleconv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
          )
        
        self.upconv2 = nn.Sequential(nn.ConvTranspose2d(256, 128, (4, 1), (2, 1), (1, 0)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        self.doubleconv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
          )
        
        self.upconv3 = nn.Sequential(nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True))
        
        self.upconv4 = nn.Sequential(nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True))
        
        self.cls = nn.Conv2d(64, self.num_classes, 3, padding=1)

    def forward(self, inputs):
        stage1, stage2, stage3 = inputs #stage1 (128, 8, 32), stage2 (256, 4, 32), stage3 (384, 2, 32)
        # import pdb
        # pdb.set_trace()
        #upscale stage3 and plus stage 2
        stage3 = self.upconv1(stage3) #384->256
        stage23 = torch.cat([stage2, stage3], dim=1) #256+256
        stage23 = self.doubleconv1(stage23) # 512->256
        
        stage23 = self.upconv2(stage23) #256->128
        stage123 = torch.cat([stage1, stage23], dim=1) #128+128
        stage123 = self.doubleconv2(stage123) #256 -> 128
        
        #upscale to origin size
        stage123 = self.upconv3(stage123)
        stage123 = self.upconv4(stage123)
        
        out = self.cls(stage123)
        
        return out
    
if __name__ == '__main__':
    stage1, stage2, stage3=torch.rand(1, 128, 8, 32), torch.rand(1, 256, 4, 32), torch.rand(1, 384, 2, 32)
    
    model=SegHeadV2(in_channels=384)
    out = model([stage1, stage2, stage3])
    
    print(out.shape)
