import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True).features.to(device).eval()

#TODO VGG19 layer를 나누기
class VGG19(nn.Module):
    def __init__(self, content_image, style_image):
        super(VGG19, self).__init__()
        self.content_image = content_image
        self.style_image = style_image

        self.content_losses = []
        self.style_losses = []








def vgg19(content_image, style_image):
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    normalization = Normalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    content_image = content_image
    style_image = style_image

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for name, layer in vgg._modules.items():
        if name in ['0', '2', '5', '10']: # Style loss를 뽑기 위한 Conv
            model.add_module('conv_{}'.format(i), layer)
            style_target = model(style_image).detach()
            style_loss = StyleLoss(style_target)
            style_losses.append(style_loss)
            model.add_module('styleloss_{}'.format(i), style_loss)

        elif name in ['7']: # content loss를 뽑기 위한 Conv
            model.add_module('conv_{}'.format(i), layer)
            content_target = model(content_image).detach()
            content_loss = ContentLoss(content_target)
            content_losses.append(content_loss)
            model.add_module('contentloss_{}'.format(i), content_loss)

            style_target = model(style_image)
            style_loss = StyleLoss(style_target)
            style_losses.append(style_loss)
            model.add_module('styleloss_{}'.format(i), style_loss)

        elif name in ['1', '3', '6', '8']:
            layer = nn.ReLU(inplace=False) # inplace=True ==> inplace=False
            model.add_module('relu_{}'.format(i), layer)

        elif name in ['4', '9']:
            model.add_module('maxpool_{}'.format(i), layer)

        elif name == '11': # 11 이상의 layer들은 필요로 하지 않기에 중단
            break
        i += 1

    return model, style_losses, content_losses

class ContentLoss(nn.Module):
    def __init__(self, content_image):
        super(ContentLoss, self).__init__()
        self.content = content_image.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.content)
        return input

class StyleLoss(nn.Module):
    def __init__(self, style_image):
        super(StyleLoss, self).__init__()
        self.style = gram_matrix(style_image).detach()

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = F.mse_loss(gram, self.style)

        return input
