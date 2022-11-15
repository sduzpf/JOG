import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import truncnorm #extra import equivalent to tf.trunc initialise
import numpy as np
import scipy.io as scio

class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class VGG_16(nn.Module):
    def __init__(self, code_len):
        super(VGG_16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-1])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.vgg16.features(x)
        x = x.view(x.size(0), -1)
        feat = self.vgg16.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class VGG_19(nn.Module):
    def __init__(self, code_len):
        super(VGG_19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg19.classifier = nn.Sequential(*list(self.vgg19.classifier.children())[:-1])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.vgg19.features(x)
        x = x.view(x.size(0), -1)
        feat = self.vgg19.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class resnet18(nn.Module):
    def __init__(self, code_len):
        super(resnet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.classifier = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.fc_encode = nn.Linear(512, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.resnet18.classifier(x)
        feat = x.view(x.size(0), -1)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class resnet50(nn.Module):
    def __init__(self, code_len):
        super(resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.classifier = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc_encode = nn.Linear(2048, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.resnet50.classifier(x)
        feat = x.view(x.size(0), -1)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


def truncated_normal_(self,tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class CNN_Text(nn.Module):

    def __init__(self, code_len, vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda=0.0001):
        super(CNN_Text, self).__init__()

        V = vocab_size
        D = embedding_dim
        C = code_len
        Ci = 1
        Co = num_filters
        self.filter_sizes = filter_sizes
        Ks = [int(filter_size) for filter_size in self.filter_sizes.split(',')]

        self.embed = nn.Embedding(V, D, padding_idx=1386)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        self.alpha = 1.0

    def forward(self, x, y):
        x = self.embed(x)  # (N, W, D)
        x = x + y * 0.00001
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        feat = self.dropout(x)  # (N, len(Ks)*Co)
        hid = self.fc1(feat)  # (N, C)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)