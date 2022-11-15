import torch
import torch.nn.functional as F  # torch.tanh
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio
from utils.metric1 import compress_wiki, compress, calculate_map, calculate_top_map
from utils.models1 import ImgNet, TxtNet, ImgNet1, VGG_16, ImgNet1, VGG_19, resnet18, resnet50, LightCNN_Text, CNN_Text
import time
import os
from utils.utils import *
import logging
from models import *
import argparse
from torchstat import stat

parser = argparse.ArgumentParser(description="ADSH demo")
parser.add_argument('--bits', default='64', type=str,help='binary code length (default: 8,12,16,24,32,48,64,96,128)')
parser.add_argument('--gpu', default='3', type=str,help='selected gpu (default: 1)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 64)')
parser.add_argument('--BETA', default=4, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LAMBDA1', default=0.1, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LAMBDA2', default=0.1, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--NUM-EPOCH', default=300, type=int, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--NUM-EPOCH1', default=300, type=int, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--NUM-EPOCH2', default=500, type=int, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR-IMG', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR-TXT', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--alpha', default=0.4, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--MOMENTUM', default=0.9, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--WEIGHT-DECAY', default=5e-4, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--ema-decay', default=0.999, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--consistency', default=100.0, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--consistency-rampup', default=30, type=int, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--NUM-WORKERS', default=4, type=int, help='number of epochs (default: 3)')
parser.add_argument('--EVAL', default= False, type=bool,help='selected gpu (default: 1)')
parser.add_argument('--EPOCH-INTERVAL', default=2, type=int, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--EVAL-INTERVAL', default=40, type=int, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--mu', default=1.4, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--ETA', default=0.5, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--INTRA', default=0.1, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--beta', default=0.3, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--lamb', default=0.3, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--MIN', default=- 0.64, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--MAX', default=0.75, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--ALPHA', default=2, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LOC_LEFT', default=- 0.62, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--SCALE_LEFT', default=0.0128, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LOC_RIGHT', default=- 0.62, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--SCALE_RIGHT', default=0.086, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--L1', default=0.2, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--L2', default=0.2, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=128, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='2,3,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-vocab-size', type=int, default=1387, help='number of each kind of kernel')

class Session:
    def __init__(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    def define_model(self, coed_length):
        # self.FeatNet_I = ImgNet1(code_len=coed_length)
        self.FeatNet_I = VGG_16(code_len=coed_length)
        # self.FeatNet_I = resnet18(code_len=coed_length)
        # VGG_19  resnet18  resnet50 VGG_16

        self.CodeNet_s1_I = ShuffleNetV2(code_len=coed_length, net_size=0.5)  # MobileNetV2
        # self.CodeNet_s1_I =MobileNetV2(code_len=coed_length)
        self.CodeNet_s1_T = CNN_Text(code_len=coed_length, vocab_size = args.vocab_size, embedding_dim = args.embed_dim,
                                          filter_sizes = args.kernel_sizes, num_filters = args.kernel_num, l2_reg_lambda=0.0001)
        # self.CodeNet_s1_T = LightCNN_Text(code_len=coed_length, vocab_size = args.vocab_size, embedding_dim = args.embed_dim,
        #                                   filter_sizes = args.kernel_sizes, num_filters = args.kernel_num, l2_reg_lambda=0.0001)

        # self.opt_s1_I = torch.optim.SGD(self.CodeNet_s1_I.parameters(), lr=args.LR_IMG, momentum=args.MOMENTUM,weight_decay=args.WEIGHT_DECAY)
        # self.opt_s1_T = torch.optim.SGD(self.CodeNet_s1_T.parameters(), lr=args.LR_TXT, momentum=args.MOMENTUM,weight_decay=args.WEIGHT_DECAY)

        self.best_it = 0
        self.best_ti = 0

        stat(self.CodeNet_s1_I, (3, 224, 224))
        # stat(self.FeatNet_I, (3, 224, 224))
        # stat(self.CodeNet_s1_T, (1, 1, 50))

        # logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        # logger.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (self.best_it, self.best_ti))
        # logger.info('MAP 500 of Image to Text: %.3f, MAP 500 of Text to Image: %.3f' % (MAP_I2T1, MAP_T2I1))
        # logger.info('--------------------------------------------------------------------')

def mkdir_multi(path):
    # 判断路径是否存在
    isExists = os.path.exists(path)

    if not isExists:
        # 如果不存在，则创建目录（多层）
        os.makedirs(path)
        print('successfully creat path！')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('path already exists！')
        return False


def _logging():
    global logger
    # logfile = os.path.join(logdir, 'log.log')
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def main():
    global logdir, args

    args = parser.parse_args()

    sess = Session()

    bits = [int(bit) for bit in args.bits.split(',')]
    for bit in bits:
        logdir = '/home/pengfei/code/hashing/work/1/mir/result/para/jdsh/resnet50/MobileNetV2/' + str(args.LR_IMG) + '/' + str(bit) + '/'
        mkdir_multi(logdir)
        _logging()
        # MobileNetV2  AlexNet
        if args.EVAL == True:
            sess.load_checkpoints()
        else:
            logger.info('--------------------------pre_train Stage--------------------------')
            sess.define_model(bit)

if __name__=="__main__":
    main()