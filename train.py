import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import argparse
import csv
from model import AVENet
from a2s_dataloader import GetVGGSound
from torch.utils.tensorboard import SummaryWriter
from inference.generate_images import get_model
from torchvision.utils import save_image,make_grid
from loss import *
import random
from torchvision.transforms.functional import to_pil_image

def get_arguments():
    parser = argparse.ArgumentParser()
    #Define Sound2Scene model
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=2048,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')


    #Define data directories
    parser.add_argument('--data_path', dest='data_path', default="./samples/training",
                        help='Path of dataset directory for train model')
    parser.add_argument("--dataset", type=str, default="vgg", choices=["vgg", "vegas"],
                        help="Dataset in which the model has been trained on.", )
    parser.add_argument("--save_path", type=str, default="./samples/output/best.pth",help="path to save trained Sound2Scene")

    #Define training settings
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--resume_epoch',default=0, type=int)

    parser.add_argument('--train', default=True, help='Train or inference')
    parser.add_argument('--warm', default=False, help='Train or inference')

    #Define image encoder and image decoder
    parser.add_argument('--root_path', default="./checkpoints", help='Train or inference')
    parser.add_argument('--model', default="icgan", help='Train or inference')
    parser.add_argument('--model_backbone', default="biggan", help='Train or inference')
    parser.add_argument('--resolution', default=128, help='Train or inference')
    parser.add_argument("--z_var", type=float, default=1.0, help="Noise variance: %(default)s)")
    parser.add_argument("--trained_dataset",type=str,default="imagenet",choices=["imagenet", "coco"],help="Dataset in which the model has been trained on.",)


    return parser.parse_args()


def showImage(args,generator,emb,gt_emb,img):
    output=None
    z = torch.empty(16,generator.dim_z).normal_(mean=0, std=args.z_var)

    emb /= torch.linalg.norm(emb, dim=-1, keepdims=True)
    gt_emb /= torch.linalg.norm(gt_emb, dim=-1, keepdims=True)

    gen_emb = generator(z.cuda(), None, emb[:16])
    gen_emb = torch.clamp(gen_emb,-1., 1.)
    gen_gt = generator(z.cuda(), None, gt_emb[:16])
    gen_gt = torch.clamp(gen_gt, -1., 1.)

    output = torch.cat((img[:8].squeeze(1), gen_gt[:8]),0)
    output = torch.cat((output,gen_emb[:8]),0)

    output = torch.cat((output,img[8:16].squeeze(1)),0)
    output = torch.cat((output, gen_gt[8:16]), 0)
    output = torch.cat((output, gen_emb[8:16]), 0)
    output = make_grid(output, normalize=True, scale_each=True, nrow=8)

    return output


def load(args, device):
    model = AVENet(args).to(device)
    if args.warm:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        for index,child in enumerate(model.children()):
            if index==0:
                num_ftrs=child.fc.in_features
                child.fc = nn.Linear(num_ftrs, 256).to(device)

        print("load_warm_start")
    return model

def load_dataset(args):
    train_dataset = GetVGGSound(args.data_path)
    test_dataset = GetVGGSound(args.data_path)


    return train_dataset, test_dataset


def validate(args, feature_extractor, generator, model, test_loader, device):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = InfoNCE_with_L2(device)

    total_loss = torch.Tensor([0])
    for step, (index, spec, emb_img, orig_img) in enumerate(test_loader):
        spec = Variable(spec).to(device)

        #extract image feature
        gt_emb, _ = feature_extractor(emb_img.squeeze(1).cuda())

        #extract audio feature
        _, audio_emb = model(spec.unsqueeze(1).float())


        loss = criterion.loss_fn(audio_emb,gt_emb)
        total_loss+=loss.item()

    return torch.mean(total_loss)

def train(args):
    #summary_writer = SummaryWriter(summary_root + args.expname)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load(args, device)
    #Define image encoder and image decoder
    ### -- Model -- ###
    suffix = (
        "_nofeataug"
        if args.resolution == 256
        and args.trained_dataset == "imagenet"
        else ""
    )
    exp_name = "%s_%s_%s_res%i%s" % (
        args.model,
        args.model_backbone,
        args.trained_dataset,
        args.resolution,
        suffix,
    )
    generator, feature_extractor = get_model(
        exp_name, args.root_path, args.model_backbone, device=device
    )

    #####define optimizer
    criterion = InfoNCE_with_L2(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    #define an dataset
    train_dataset, test_dataset = load_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    min_loss = torch.tensor(100).detach().cpu()

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = torch.tensor(0)
        epoch_iter = 0

        for step, (index, spec, emb_img, orig_img) in enumerate(train_loader):
            spec = Variable(spec).to(device)

            #extract image feature
            gt_emb, _ = feature_extractor(emb_img.squeeze(1).cuda())

            #extract audio feature
            _, audio_emb = model(spec.unsqueeze(1).float())


            loss = criterion.loss_fn(audio_emb,gt_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            epoch_iter = epoch_iter + 1
            epoch_train_loss=epoch_train_loss+loss

            print('[epoch: %d] iter_train_loss: %.3f' % (epoch + 1, loss.detach().item() ), end='\r')

        #validate
        model.eval()
        val_loss = validate(args, feature_extractor, generator, model, test_loader, device)
        model.train()

        if val_loss<min_loss:
            min_loss=val_loss
            torch.save(model.state_dict(), args.save_path)


def main():
    random_seed=1234
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = get_arguments()
    train(args)

if __name__=='__main__':
    main()