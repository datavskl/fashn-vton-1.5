"""Experimental supervised mobile VTON student. No pretrained weights included."""
import torch
from torch import nn

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())

class EdgeVTON(nn.Module):
    """Inputs: RGB person, RGB flat-lay garment, 3 category maps; all spatially aligned."""
    def __init__(self):
        super().__init__()
        self.e1=block(9,32)
        self.e2=block(32,64)
        self.e3=block(64,128)
        self.mid=block(128,256)
        self.pool=nn.AvgPool2d(2)
        self.up=nn.Upsample(scale_factor=2,mode="nearest")
        self.d3=block(384,128)
        self.d2=block(192,64)
        self.d1=block(96,32)
        self.out=nn.Conv2d(32,3,1)
    def forward(self,x):
        a=self.e1(x)
        b=self.e2(self.pool(a))
        c=self.e3(self.pool(b))
        z=self.mid(self.pool(c))
        z=self.d3(torch.cat((self.up(z),c),1))
        z=self.d2(torch.cat((self.up(z),b),1))
        z=self.d1(torch.cat((self.up(z),a),1))
        return torch.sigmoid(self.out(z))
