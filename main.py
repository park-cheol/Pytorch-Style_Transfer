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
from model import *
# 모델 불러오기
""" 불필요
model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__")
                     and callable(models.__dict__[name]))
"""
# .__dict__: 어떤 속성이 있는지 확인, 그 중에서 소문자, __으로 시작하지 않은것, 호출 가능한 것만 모아놓음
parser = argparse.ArgumentParser(description='PyTorch Image Transfer Using Cnn')
""" vgg19 고정으로 쓸 것
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg19',
                    # metavar: usage 메시지를 출력할 때 표시할 메타변수이름 지정
                    choices=model_names,  # 인자를 허용되는 값의 컨테이너 --> 이 안에서 선택
                    help='model architecture: ' +
                         '|'.join(model_names) +
                         ' (default: vgg19)')
"""
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

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

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

    # CUDA
    if not device == 'cuda':  # False
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)  # 특정 GPU 선택 [0]

    # Image Loading
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    content_img = utils.load_image(args.content_image, args.image_size)
    content_img = transform(content_img).to(device)
    content_img = content_img.repeat(1, 1, 1, 1).to(device) # 배치 차원 추가
    style_img = utils.load_image(args.style_image, args.image_size)
    style_img = transform(style_img).to(device)
    style_img = style_img.repeat(1, 1, 1, 1).to(device) # 배치 차원 추가

    input_img = torch.empty_like(content_img).uniform_(0, 1).to(device)


    vgg, content_losses, style_losses = vgg19(content_img, style_img)

    content_score = 0
    style_score = 0
    for epoch in range(args.epochs):
        # train
        train(epoch, vgg, content_losses, style_losses, content_score, style_score)



def train(epoch, vgg, input_img, content_losses, style_losses, content_score, style_score):
    optimizer = torch.optim.Adam([input_img.requires_grad_()])
    optimizer.zero_grad()
    input_img.data.clamp_(0, 1)
    vgg(input_img)

    for cl in content_losses:
        content_score += cl.loss

    for sl in style_losses:
        style_score += sl.loss

    style_score *= args.style_weight / args.content_weight
    loss = content_score + style_score
    loss.backward()

    optimizer.step()

    if epoch % 100 == 0: # log
        print(f"[ step:{epoch} / content loss: {content_score.item()} / style loss: {style_score.item()}")

    if epoch % 5000 == 0: # save image per 5000
        torchvision.utils.save_image(input_img.cpu().detach()[0], 'image/output1_%s.png' % (epoch))
        print("save image")

if __name__ == '__main__':
    main()
# import로 호출되어 사용 시 파일 명이 뜸 / 처음 실행한 파일은 __main__