import torch
import torch.nn as nn 


class RealTimeSRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on 
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """
    def __init__(self, 
                 num_channels, 
                 num_feats, 
                 num_blocks,
                 upscale) -> None:
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(num_channels, num_feats, 3, padding=1)
        )
        
        body = []
        for i in range(num_blocks):
            body.append(nn.Conv2d(num_feats, num_feats, 3, padding=1))
            if i < num_blocks -1:
                body.append(nn.ReLU(True))
                
        self.body = nn.Sequential(*body)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale)            
        )
    
    def forward(self, x):
        res = self.head(x)
        out = self.body(res)
        out = self.upsample(res + out)        
        return out
    
    
    
def rtsrn(scale):
    model = RealTimeSRNet(num_channels=3, num_feats=64, num_blocks=5, upscale=scale)
    return model
