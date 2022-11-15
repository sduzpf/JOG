import torch
import torch.nn.functional as F  # torch.tanh
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio
from utils.metric1 import compress_wiki, compress, calculate_map, calculate_top_map, NDCG
import time
import os
from utils.utils import *
import logging
from models import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="ADSH demo")
parser.add_argument('--bits', default='64', type=str,help='binary code length (default: 8,12,16,24,32,48,64,96,128)')
parser.add_argument('--LR-IMG', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--model1', default='AlexNet', type=str,help='binary code length (default: AlexNet,VGG_16,resnet50,VGG_19,resnet18)')
parser.add_argument('--model2', default='ShuffleNetV2', type=str,help='binary code length (default: ShuffleNetV2,MobileNetV2)')
parser.add_argument('--k_list', default='5,10', type=str,help='binary code length (default: 8,12,16,24,32,48,64,96,128)')

class Session:
    def __init__(self):

        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)

        self.best_it = 0
        self.best_ti = 0

    def eval(self,bit):
        logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).

        result_DIR = '/home/pengfei/code/hashing/work/1/mir/result/Textcnn1/dual/' + args.model1 + '/'+ args.model2  + '/' + str(args.LR_IMG) + '/'+ str(
                bit) + '/' + 'code.mat'

        result_set = sio.loadmat(result_DIR)

        re_BI = np.array(result_set['re_BI'], dtype=np.float)
        re_BT = np.array(result_set['re_BT'], dtype=np.float)
        qu_BI = np.array(result_set['qu_BI'], dtype=np.float)
        qu_BT = np.array(result_set['qu_BT'], dtype=np.float)
        re_L = np.array(result_set['re_L'], dtype=np.float)
        qu_L = np.array(result_set['qu_L'], dtype=np.float)

        k_lists = [int(k_list) for k_list in args.k_list.split(',')]
        for k_list in k_lists:
            self.calculate_NDCG = NDCG(k_list)
            NDCG_I2T = self.calculate_NDCG._get_target(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
            NDCG_T2I = self.calculate_NDCG._get_target(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
            # NDCG_I2T1 = self.calculate_NDCG._get_target(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=500)  # 500
            # NDCG_T2I1 = self.calculate_NDCG._get_target(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=500)
            logger.info('NDCG @k: %.3f, Image to Text: %.3f, Text to Image: %.3f' % (k_list,NDCG_I2T, NDCG_T2I))
            # logger.info('NDCG @k: %.3f, Image to Text: %.3f,Text to Image: %.3f' % (NDCG_I2T1, NDCG_T2I1))
            logger.info('--------------------------------------------------------------------')

        # MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        # MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        # logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))


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
        logdir = '/home/pengfei/code/hashing/work/1/mir/result/valuate/dual/' + args.model1 + '/'+ args.model2  + '/'  + str(args.LR_IMG) + '/' + str(bit) + '/'
        #   VGG_16  resnet50   ShuffleNetV2  resnet18
        mkdir_multi(logdir)
        _logging()
        sess.eval(bit)

if __name__=="__main__":
    main()