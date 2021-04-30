import numpy
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_image(filename, size=None):
    img = Image.open(filename).convert('RGB') # RGB 전환
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS) #TODO Interpolate
    return img

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b * ch, h * w)
    gram = torch.mm(features, features.t())
    # mm -> 행렬곱 (비슷한 것: element-wise/아다마르곱==> element끼리 곱)
    gram = gram.div(b * ch * h * w)
    return gram

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization,  self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

