import torch
import torch.nn as nn
import numpy as np



class L2_Loss():
    def __init__(self,device):
        self.criterion = nn.MSELoss()
        self.device=device
    def loss_fn(self,aud_feature,img_feature):
        loss = self.criterion(img_feature, aud_feature)

        return loss

class L1_Loss():
    def __init__(self,device):
        self.criterion = nn.L1Loss()
        self.device=device
    def loss_fn(self,aud_feature,img_feature):
        loss = self.criterion(img_feature, aud_feature)

        return loss


class InfoNCE_Loss():
    def __init__(self,device):
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_aud = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07))
        self.device = device

    def loss_fn(self, audio_features, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        # logit_scale = self.logit_scale.exp()
        logit_scale=1
        logits_per_image = logit_scale * image_features @ audio_features.t()
        logits_per_aud = logits_per_image.t()

        ground_truth = torch.arange(audio_features.shape[0], dtype=torch.long, device=self.device)
        total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_aud(logits_per_aud, ground_truth)) / 2


        return total_loss

class InfoNCE_with_L2():
    def __init__(self,device):
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_aud = nn.CrossEntropyLoss()

        # self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * 1)
        self.device = device

    def loss_fn(self, audio_features, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)


        cdist_per_image = torch.cdist(image_features, audio_features, p=2)*self.logit_scale
        # cdist_per_image = (torch.diag(torch.matmul(image_features,image_features.transpose(0,1)),0)+torch.diag(torch.matmul(audio_features,audio_features.transpose(0,1)),0)-2*torch.matmul(image_features,audio_features.transpose(0,1)))**(1/2)
        cdist_per_aud = cdist_per_image.t()

        ground_truth = torch.arange(audio_features.shape[0], dtype=torch.long, device=self.device)
        loss1= self.loss_img(-cdist_per_image, ground_truth)
        loss2=self.loss_aud(-cdist_per_aud, ground_truth)
        total_loss = (loss1+loss2)/2
        #total_loss = (self.loss_img(-cdist_per_image, ground_truth) + self.loss_aud(-cdist_per_aud, ground_truth)) / 2



        return total_loss
