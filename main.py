import argparse
import random
import numpy
import PIL
import matplotlib as plt
import warnings
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import utils

# 모델 불러오기
model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__")
                     and callable(models.__dict__[name]))
# .__dict__: 어떤 속성이 있는지 확인, 그 중에서 소문자, __으로 시작하지 않은것, 호출 가능한 것만 모아놓음
parser = argparse.ArgumentParser(description='PyTorch Image Transfer Using Cnn')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg19',
                    # metavar: usage 메시지를 출력할 때 표시할 메타변수이름 지정
                    choices=model_names,  # 인자를 허용되는 값의 컨테이너 --> 이 안에서 선택
                    help='model architecture: ' +
                         '|'.join(model_names) +
                         ' (default: vgg19)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--style-image', type=str, default='image/style/style1.jpg',
                    help='path to style-image')
parser.add_argument('--content-image', type=str, default='image/content/content1.jpg',
                    help='path to content-image')
parser.add_argument('--image-size', type=int, default=512,
                    help='size of style images, default is 512 X 512')
parser.add_argument('--content-weight', type=float, default=1e5,
                    help='weight for content-loss, default is 1e5')
parser.add_argument('--style-weight', type=float, default=1e10,
                    help='weight for style-loss, default is 1e10')
parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true',
                    help='use pre-trained model')
# dest: parser_args()가 반환하는 객체에 추가될 attribute의 이름( .pretrained)
# action='store_true'(stroe_false): 인자를 적으면 해당 인자에 True나 False 저장
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='gpu id to use. ')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parser.parse_args()

    if args.seed is not None:  # reproductive
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'cudnn.deterministic는 학습을 상당히 느리게 하고, '
                      'checkpoints로 재시작 시 예상치 못한 결과를 볼 수 도 있다. ')
        # 권장: 실험하는 초기 단계가 아닌 모델과 코드를 배포해야 하는 단게에서 사용
    if args.gpu is not None:
        warnings.warn('특정 GPU-id를 선택. 이는 data parallelism을 사용 할 수 없습니다. ')
    # data parallelism을 사용하여 여러 GPU 사용[ nn.DataParallel(model) ]
    # torch.cuda.device_count() > 1 ==> GPU 몇 개인지 파악, 1개 초과만
    # TODO 데이터병렬처리, 분산학습

    ngpus_per_node = torch.cuda.device_count()  # 현재 컴퓨터로 1
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu  # TODO 2번 재선언 해주는 이유

    if args.gpu is not None:
        print("use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}".format(args.arch))
        model = models.__dict__[args.arch]()

    # CUDA
    if not device == 'cuda':  # False
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)  # 특정 GPU 선택 [0]
        model = model.cuda(args.gpu)  # CUDA 할당

    # Image Loading
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    content_img = utils.load_image(args.content_image, args.image_size)
    content_img = transform(content_img).to(device)
    content_img = content_img.repeat(1, 1, 1, 1).to(device)
    style_img = utils.load_image(args.style_image, args.image_size)
    style_img = transform(style_img).to(device)
    style_img = style_img.repeat(1, 1, 1, 1).to(device)

    noise_img = torch.empty_like(content_img).uniform_(0, 1).to(device)

    cnn = model.features.to(device).eval()

    vgg, content_losses, style_losses = losses(cnn, content_img, style_img)

    # optimizer
    optimizer = torch.optim.Adam([noise_img.requires_grad_()])

    # train
    print("[ start ]")
    for epoch in range(args.epochs):
        noise_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        vgg(noise_img)
        content_score = 0
        style_score = 0

        for content in content_losses:
            content_score += content.loss
        for style in style_losses:
            style_score += style.loss
        content_score *= args.content_weight
        style_score *= args.style_weight
        loss = content_score + style_score
        loss.backward()

        if epoch % 100 == 0:
            print(f"[ Step: {epoch} / content loss: {content_score.item()} / style loss: {style_score.item()}")
        if epoch % 5000 == 0:
            torchvision.utils.save_image(noise_img.cpu().detach()[0], 'image/output/output_%s.png' % (epoch))
            print("Save image")

        optimizer.step()

def losses(cnn, content_img, style_img):
    cnn = copy.deepcopy(cnn)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = utils.Normalization(mean, std).to(device)

    content_layers = ['conv_5']
    style_layers = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_15']
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # 설정한 content layer까지의 결과를 이용해 content loss를 계산
        if name in content_layers:
            target_feature = model(content_img).detach()
            content_loss = utils.ContentLoss(target_feature)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # 설정한 style layer까지의 결과를 이용해 style loss를 계산
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = utils.StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 마지막 loss 이후의 레이어는 사용하지 않도록
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], utils.ContentLoss) or isinstance(model[i], utils.StyleLoss):
            break

    model = model[:(i + 1)]
    return model, content_losses, style_losses


if __name__ == '__main__':
    main()
# import로 호출되어 사용 시 파일 명이 뜸 / 처음 실행한 파일은 __main__