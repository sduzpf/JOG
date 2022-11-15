import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from torch.autograd import Function
# import torchvision.models as models
# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)

class CNNNet(nn.Module):
    def __init__(self, model_name, code_length, pretrained=True):
        super(CNNNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'vgg11'

        if model_name == "vgg16":
            original_model = models.vgg16(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'vgg16'

        if model_name == "vgg19":
            original_model = models.vgg19(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'vgg19'

        if model_name == 'resnet18':
            original_model = models.resnet18(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'resnet18'

        if model_name == 'resnet50':
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'resnet50'

        if model_name == 'resnet101':
            original_model = models.resnet101(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'resnet101'

        if model_name == 'resnext':
            original_model = models.resnext50_32x4d(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'resnext'

        if model_name == 'densenet121':
            original_model = models.densenet121(pretrained)
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'densenet121'


        if model_name == 'inception_v3':
            original_model = models.inception_v3(pretrained)
            # self.features = original_model.features
            # original_model.Auxlogits.fc = nn.Linear(768, code_length)
            original_model.AuxLogits.fc = nn.Linear(original_model.AuxLogits.fc.in_features, code_length)
            original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 全局池化
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'inception_v3'

        if model_name == 'SqueezeNet':
            original_model = models.squeezenet1_0(pretrained) # 1_0/1-1
            self.features = original_model.features
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'SqueezeNet'

        if model_name == 'mobilenet':
            original_model = models.mobilenet_v2(pretrained)
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'mobilenet'

        if model_name == 'ShuffleNet':
            original_model = models.shufflenet_v2_x1_0(pretrained)
            # original_model.aux_logits=False
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'ShuffleNet'

        if model_name == 'WideResNet':
            original_model = models.wide_resnet50_2(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'WideResNet'

        if model_name == 'MNASNet':
            original_model = models.mnasnet0_5(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'MNASNet'

        if model_name == 'googlenet':
            original_model = models.googlenet(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'googlenet'


    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
        if self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
        if self.model_name == 'vgg19':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet18':
            f = f.view(f.size(0), -1)
            # f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'resnet50':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet101':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnext':
            f = f.view(f.size(0), -1)
        if self.model_name == 'densenet121':
            f = f.view(f.size(0), -1)
        if self.model_name == 'inception_v3':
            f =self.avg_pool(f)
            f = f.view(-1, f.shape[1] * f.shape[2] * f.shape[3])
        if self.model_name == 'SqueezeNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'mobilenet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'ShuffleNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'WideResNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'MNASNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'googlenet':
            f = f.view(f.size(0), -1)
        # y = self.classifier(f)
        return f

class BufferNet(nn.Module):
    def __init__(self, model_name, code_length, queue_size, pretrained=True):
        super(BufferNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg11'

        if model_name == "vgg16":
            original_model = models.vgg16(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            # cl1 = nn.Linear(25088, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg16'

        if model_name == "vgg19":
            original_model = models.vgg19(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            # cl1 = nn.Linear(25088, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg19'

        if model_name == 'resnet18':
            original_model = models.resnet18(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet18'

        if model_name == 'resnet50':
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet50'

        if model_name == 'resnet101':
            original_model = models.resnet101(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet101'

        if model_name == 'resnext':
            original_model = models.resnext50_32x4d(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnext'

        if model_name == 'densenet121':
            original_model = models.densenet121(pretrained)
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(50176, code_length),
                nn.Tanh()
            )
            self.model_name = 'densenet121'

        if model_name == 'inception_v3':
            original_model = models.inception_v3(pretrained)
            # self.features = original_model.features
            # original_model.Auxlogits.fc = nn.Linear(768, code_length)
            original_model.AuxLogits.fc = nn.Linear(original_model.AuxLogits.fc.in_features, code_length)
            original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 全局池化
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'inception_v3'

        if model_name == 'SqueezeNet':
            original_model = models.squeezenet1_0(pretrained) # 1_0/1-1
            self.features = original_model.features
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            # self.classifier = nn.Sequential(
            #     nn.Linear(256, code_length),
            #     nn.Tanh()
            # )
            # Final convolution is initialized differently from the rest
            final_conv = nn.Conv2d(512, code_length, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.model_name = 'SqueezeNet'

        if model_name == 'mobilenet':
            original_model = models.mobilenet_v2(pretrained)
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(62720, code_length),
                # nn.Linear(1280, code_length),
                nn.Tanh()
            )
            self.model_name = 'mobilenet'

        if model_name == 'ShuffleNet':
            original_model = models.shufflenet_v2_x1_0(pretrained)
            # original_model.aux_logits=False
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(86528, code_length),
                nn.Tanh()
            )
            self.model_name = 'ShuffleNet'

        if model_name == 'WideResNet':
            original_model = models.wide_resnet50_2(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'WideResNet'

        if model_name == 'MNASNet':
            original_model = models.mnasnet0_5(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(62720, code_length),
                nn.Tanh()
            )
            self.model_name = 'MNASNet'

        if model_name == 'googlenet':
            original_model = models.googlenet(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(1024, code_length),
                nn.Tanh()
            )
            self.model_name = 'googlenet'

        # queue_name = "queue" + name
        #if name== "noise":
        self.register_buffer("queue", torch.randn(queue_size, code_length))
        # self.register_buffer("queue1", torch.randn(queue_size, code_length))

        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # self.register_buffer("queue", torch.randn(code_length, queue_size))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size
        # print(self.queue.size())

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        # T=0.1
        # print(keys.T)
        # replace the keys at ptr (dequeue and enqueue)
        # print(keys.size())
        # print('------------------------')
        # print(self.queue.size())
        self.queue[ptr:ptr + batch_size, :] = keys  #.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, x, name="clean"):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
        if self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
        if self.model_name == 'vgg19':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet18':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet50':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet101':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnext':
            f = f.view(f.size(0), -1)
        if self.model_name == 'densenet121':
            f = f.view(f.size(0), -1)
        if self.model_name == 'inception_v3':
            f =self.avg_pool(f)
            f = f.view(-1, f.shape[1] * f.shape[2] * f.shape[3])
        if self.model_name == 'SqueezeNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'mobilenet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'ShuffleNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'WideResNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'MNASNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'googlenet':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)

        self.history = self.queue.clone().detach()
           # dequeue and enqueue
        if name == 'record':
           self._dequeue_and_enqueue(y)

        return y, self.history

class BufferNet1(nn.Module):
    def __init__(self, model_name, code_length, queue_size, pretrained=True):
        super(BufferNet1, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg11'

        if model_name == "vgg16":
            original_model = models.vgg16(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            # cl1 = nn.Linear(25088, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg16'

        if model_name == "vgg19":
            original_model = models.vgg19(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            # cl1 = nn.Linear(25088, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg19'

        if model_name == 'resnet18':
            original_model = models.resnet18(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet18'

        if model_name == 'resnet50':
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet50'

        if model_name == 'resnet101':
            original_model = models.resnet101(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet101'

        if model_name == 'resnext':
            original_model = models.resnext50_32x4d(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnext'

        if model_name == 'densenet121':
            original_model = models.densenet121(pretrained)
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(50176, code_length),
                nn.Tanh()
            )
            self.model_name = 'densenet121'

        if model_name == 'inception_v3':
            original_model = models.inception_v3(pretrained)
            # self.features = original_model.features
            # original_model.Auxlogits.fc = nn.Linear(768, code_length)
            original_model.AuxLogits.fc = nn.Linear(original_model.AuxLogits.fc.in_features, code_length)
            original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 全局池化
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'inception_v3'

        if model_name == 'SqueezeNet':
            original_model = models.squeezenet1_0(pretrained) # 1_0/1-1
            self.features = original_model.features
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            # self.classifier = nn.Sequential(
            #     nn.Linear(256, code_length),
            #     nn.Tanh()
            # )
            # Final convolution is initialized differently from the rest
            final_conv = nn.Conv2d(512, code_length, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.model_name = 'SqueezeNet'

        if model_name == 'mobilenet':
            original_model = models.mobilenet_v2(pretrained)
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(62720, code_length),
                # nn.Linear(1280, code_length),
                nn.Tanh()
            )
            self.model_name = 'mobilenet'

        if model_name == 'ShuffleNet':
            original_model = models.shufflenet_v2_x1_0(pretrained)
            # original_model.aux_logits=False
            self.features = original_model.features
            # self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(86528, code_length),
                nn.Tanh()
            )
            self.model_name = 'ShuffleNet'

        if model_name == 'WideResNet':
            original_model = models.wide_resnet50_2(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'WideResNet'

        if model_name == 'MNASNet':
            original_model = models.mnasnet0_5(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(62720, code_length),
                nn.Tanh()
            )
            self.model_name = 'MNASNet'

        if model_name == 'googlenet':
            original_model = models.googlenet(pretrained)
            # original_model.aux_logits=False
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(1024, code_length),
                nn.Tanh()
            )
            self.model_name = 'googlenet'

        # queue_name = "queue" + name
        # self.register_buffer("queue1", torch.randn(code_length, queue_size))
        self.register_buffer("queue1", torch.randn(queue_size, code_length))
        self.queue1 = nn.functional.normalize(self.queue1, dim=0)
        self.register_buffer("queue_ptr1", torch.zeros(1, dtype=torch.long))

        # self.register_buffer("queue", torch.randn(code_length, queue_size))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr1)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[ptr:ptr + batch_size, :] = keys  #.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr1[0] = ptr


    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
        if self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
        if self.model_name == 'vgg19':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet18':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet50':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet101':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnext':
            f = f.view(f.size(0), -1)
        if self.model_name == 'densenet121':
            f = f.view(f.size(0), -1)
        if self.model_name == 'inception_v3':
            f =self.avg_pool(f)
            f = f.view(-1, f.shape[1] * f.shape[2] * f.shape[3])
        if self.model_name == 'SqueezeNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'mobilenet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'ShuffleNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'WideResNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'MNASNet':
            f = f.view(f.size(0), -1)
        if self.model_name == 'googlenet':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)

        self.history = self.queue1.clone().detach()
        # dequeue and enqueue
        self._dequeue_and_enqueue(y)

        return y, self.history


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #              for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=True)

    # output = torch.cat(tensor, dim=0)
    # return output



class CNNExtractNet(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CNNExtractNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
            )
            self.model_name = 'vgg11'
        if model_name == "resnet50":
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'resnet50'


    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)

        if self.model_name == "resnet50":
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):

    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.tanh(out)

        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, eps):
        # x = self.encoder(x)
        # x = self.decoder(x)
        # return x

        noise_x = self.encoder(x)
        noise_x = self.decoder(noise_x)
        noise = noise_x * eps
        noise = torch.clamp(x + noise, 0, 1)
        atkdata = ReverseLayerF.apply(noise, 1)

        return noise_x, atkdata

class Domaincl(nn.Module):
    def __init__(self,code_length, pretrained=True):
        super(Domaincl, self).__init__()
        original_model = models.alexnet(pretrained)
        self.features = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl2 = nn.Linear(4096, 1024)
        if pretrained:
            cl1.weight = original_model.classifier[1].weight
            cl1.bias = original_model.classifier[1].bias
            # cl2.weight = original_model.classifier[4].weight
            # cl2.bias = original_model.classifier[4].bias

        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2)
        )

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(3, 64, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 128, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 512, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(2048, 2),
        #     nn.Tanh()
        # )
    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class GCN(nn.Module):
    def __init__(self, code_len):
        super(GCN, self).__init__()

        self.gconv1 = nn.Linear(4096, 4096)
        self.BN1 = nn.BatchNorm1d(4096)
        self.act1 = nn.ReLU()

        self.gconv3 = nn.Linear(4096, code_len)
        self.BN3 = nn.BatchNorm1d(code_len)
        self.act3 = nn.Tanh()

        self.fc = nn.Linear(code_len, code_len)

    def forward(self, x, in_affnty):
        out = self.gconv1(x)
        out = in_affnty.mm(out)
        out = self.BN1(out)
        out = self.act1(out)

        # block 3
        out = self.gconv3(out)
        out = in_affnty.mm(out)
        out = self.BN3(out)

        out = torch.tanh(out)
        return out

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None