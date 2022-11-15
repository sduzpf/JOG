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
        # Construct nn.Module superclass from the derived classs MultibranchLeNet
        super(ImgNet, self).__init__()
        # Construct MultibranchLeNet architecture
        self.conv1 = nn.Sequential()
        self.conv1.add_module('c1_conv', nn.Conv2d(3, 32, kernel_size=5))
        self.conv1.add_module('c1_relu', nn.ReLU(True))
        self.conv1.add_module('c1_pool', nn.MaxPool2d(2))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('c2_conv', nn.Conv2d(32, 48, kernel_size=5))
        self.conv2.add_module('c2_relu', nn.ReLU(True))
        self.conv2.add_module('c2_pool', nn.MaxPool2d(2))

        self.feature_classifier = nn.Sequential()
        self.feature_classifier.add_module('f_fc1', nn.Linear(48 * 53 * 53, 100))
        # self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu1', nn.ReLU(True))
        self.feature_classifier.add_module('f_fc2', nn.Linear(100, 100))
        # self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu2', nn.ReLU(True))
        self.feature_classifier.add_module('f_fc3', nn.Linear(100, code_len))

        self.alpha = 1.0

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        feat = out.view(out.size(0), -1)
        hid = self.feature_classifier(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class ImgNet1(nn.Module):
    def __init__(self, code_len):
        super(ImgNet1, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
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
        # code = F.tanh(self.alpha * hid)
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
        # code = F.tanh(self.alpha * hid)
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
        # feat = self.resnet18.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
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
        # feat = self.resnet50.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
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
        # code = F.tanh(self.alpha * hid)
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

class LightCNN_Text(nn.Module):

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    can refer to https://tieba.baidu.com/p/5707172665 for explain the kernel size settings for the conv2d
    """
    def __init__(self, code_len, vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda=0.0001):
        super(LightCNN_Text, self).__init__()
        self.embedding_size = 300
        self.filter_sizes = filter_sizes
        self.layer =[]
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters

        self.embed = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=1386)
        # self.embedding = nn.Embedding(1387, 128, padding_idx=1386)

        filter_sizes1 = [int(filter_size) for filter_size in self.filter_sizes.split(',')]
        # print(filter_sizes1)
        for i, filter_size in enumerate(filter_sizes1):
            self.layer = self._make_layer(int(filter_size), self.num_filters)
            if i == 0:
               self.convs = nn.ModuleList([self.layer])
            else:
               self.convs.append(self.layer)
        # self.convs = nn.ModuleList([self.layer[i] for i in self.filter_sizes])

        # self.fc1 = nn.Linear((len(self.filter_sizes)-2) * self.embedding_size, 512)
        self.fc1 = nn.Linear((len(self.filter_sizes)-2) * self.num_filters, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, filter_size, num_filters):
        layers = []
        layers.append(nn.BatchNorm2d(1))

        # +Dialated Conv atrous_conv2d
        if filter_size == 5:
            # layers.append(nn.Conv2d(1, 1, kernel_size=3, bias=False, dilation=2))
            # dilation calculate https: // blog.csdn.net / gshgsh1228 / article / details / 106053886
            layers.append(nn.Conv2d(1, 1, kernel_size=3, bias=False, dilation=2))
            # layers.append(nn.Conv2d(1, 1, kernel_size=(3,self.embedding_dim), bias=False, dilation=2))
            # layers.append(nn.Conv2d(1, num_filters, kernel_size=3, bias=False, dilation=3))

            # Separable Conv/depth
            layers.append(nn.Conv2d(1, 1, kernel_size=3, bias=False))
            # layers.append(nn.Conv2d(1, 1, kernel_size=(3,self.embedding_dim), bias=False))
            # layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, bias=False))
        else:
            layers.append(nn.Conv2d(1, 1, kernel_size=filter_size, bias=False))
            # layers.append(nn.Conv2d(1, 1, kernel_size=(filter_size,self.embedding_dim), bias=False))
            # ksize_1 = [1, 32 - filter_size + 1, 1, 1]
            # layers.append(nn.Conv2d(1, num_filters, kernel_size=filter_size, bias=False))
        # Pointwise Convolution Layer
        # layers.append(nn.Conv2d(num_filters, self.embedding_size, kernel_size=1, bias=False))
        layers.append(nn.Conv2d(1, num_filters, kernel_size=1, bias=False))
        # Batch Normalzation
        layers.append(nn.BatchNorm2d(num_filters))
        # layers.append(nn.Conv2d(self.embedding_size, self.embedding_size, kernel_size=1, stride=1, padding=1, bias=False))
        # layers.append(nn.BatchNorm2d(self.embedding_size))
        # Apply nonlinearity
        layers.append(nn.LeakyReLU(inplace=True))
        # layers.append(nn.MaxPool2d(ksize_1))
        # layers.append(F.max_pool2d(ksize_1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = Variable(torch.LongTensor(x))
        # print(x)
        x = x.long()
        x = x.squeeze(1)
        x = x.squeeze(1)
        x = self.embed(x)  # (N, W, D)
        # print(x.size())
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # print(x.size())
        # x = [F.max_pool2d(conv(x)) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [conv(x).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        # print(x[0].size())
        # print(x[1].size())
        # print(x[2].size())
        x = [F.max_pool2d(i, (i.size(2),i.size(3))).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # x = [i.squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # x = torch.cat(x, 3)
        # print(x[0].size())
        x = torch.cat(x, 1)
        # print(x.size())

        # x = x.view(x.size(0), [-1, self.num_filters * 3])
        x = x.view(x.size(0), -1)
        # x = torch.reshape(x, [-1, self.num_filters * 3])
        # print(x.size())

        x = self.dropout(x)  # (N, len(Ks)*Co)

        feat = self.fc1(x)  # (N, C)
        hid = self.fc2(feat)  # (N, C)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code


    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class CNN_Text(nn.Module):

    def __init__(self, code_len, vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda=0.0001):
        super(CNN_Text, self).__init__()

        V = vocab_size
        D = embedding_dim
        C = code_len
        Ci = 1
        Co = num_filters
        # Ks = filter_sizes
        self.filter_sizes = filter_sizes
        Ks = [int(filter_size) for filter_size in self.filter_sizes.split(',')]

        # self.embed = nn.Embedding(V, D)
        self.embed = nn.Embedding(V, D, padding_idx=1386)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(len(Ks) * Co, C)

        self.fc1 = nn.Linear(len(Ks) * Co, 4096)
        self.fc2 = nn.Linear(4096, C)
        self.alpha = 1.0

        # if self.args.static:
        #     self.embed.weight.requires_grad = False

        # weight_path = '/home/pengfei/code/hashing/work/1/mir/models/pre_trained/mirflickr25k-txts-w2v-id-50len.mat'
        # embedding_weights = scio.loadmat(weight_path)
        # embedding_weights = np.array(embedding_weights)
        # embedding_weights = torch.Tensor(embedding_weights)
        # self.embed.weight.data.copy_(torch.from_numpy(embedding_weights))
        # # self.embed.load_state_dict(embedding_weights)
        # self.embed.weight.requires_grad = False

        # self.embed.weight.requires_grad = False

    def forward(self, x):
        # print(x)
        x = x.long()
        x = x.squeeze(1)
        x = x.squeeze(1)
        x = self.embed(x)  # (N, W, D)
        # print(x)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = self.fc1(x)  # (N, C)
        # return logit

        feat = self.fc1(x)  # (N, C)
        hid = self.fc2(feat)  # (N, C)
        code = torch.tanh(self.alpha * hid)
        # print(code)
        return feat, hid, code


    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)