import torch
import torch.nn.functional as F  # torch.tanh
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio
from utils.metric import compress
import utils.datasets_mir as datasets_mir
from utils.models import ImgNet, VGG_19, CNN_Text
import time
import os
from utils.utils import *
import logging
from models import *
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="JOG demo")
parser.add_argument('--bits', default='32', type=str,help='binary code length (default: 8,12,16,24,32,48,64,96,128)')
parser.add_argument('--gpu', default='1', type=str,help='selected gpu (default: 1)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 64)')
parser.add_argument('--BETA', default=0.9, type=float, help='hyper-parameter for simialrity matrix construction (default: 0.9)')
parser.add_argument('--NUM-EPOCH1', default=40, type=int, help='EPOCH for warmup(default: 40)')
parser.add_argument('--NUM-EPOCH2', default=260, type=int, help='EPOCH for post training (default: 260)')
parser.add_argument('--LR-IMG', default=0.01, type=float, help='hyper-parameter: learning rate for image (default: 10**-2)')
parser.add_argument('--LR-TXT', default=0.01, type=float, help='hyper-parameter: learning rate for text (default: 10**-2)')
parser.add_argument('--alpha', default=0.8, type=float, help='hyper-parameter for simialrity matrix construction (default: 0.8)')
parser.add_argument('--MOMENTUM', default=0.9, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--WEIGHT-DECAY', default=5e-4, type=float, help='hyper-parameter: weight decay (default: 10**-3)')
parser.add_argument('--ema-decay', default=0.999, type=float, help='hyper-parameter: ema decay (default: 10**-3)')
parser.add_argument('--NUM-WORKERS', default=4, type=int, help='number of works (default: 4)')
parser.add_argument('--LAMBDA1', default=0.1, type=float, help='hyper-parameter: balancing loss items (default: 10**-1)')
parser.add_argument('--LAMBDA2', default=0.1, type=float, help='hyper-parameter: balancing loss items  (default: 10**-1)')
parser.add_argument('--EVAL', default= False, type=bool,help='')
parser.add_argument('--EPOCH-INTERVAL', default=2, type=int, help='')
parser.add_argument('--EVAL-INTERVAL', default=100, type=int, help='interval for evaluation (default: 100)')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=128, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-vocab-size', type=int, default=1387, help='vocabulary size')

class Session:
    def __init__(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)

        self.train_dataset = datasets_mir.MIRFlickr(train=True, transform=datasets_mir.mir_train_transform)
        self.test_dataset = datasets_mir.MIRFlickr(train=False, database=False, transform=datasets_mir.mir_test_transform)
        self.database_dataset = datasets_mir.MIRFlickr(train=False, database=True,
                                                       transform=datasets_mir.mir_test_transform)
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=args.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=args.NUM_WORKERS)

        self.best_it = 0
        self.best_ti = 0

    def define_model(self, coed_length):
        self.FeatNet_I = VGG_19(code_len=coed_length)

        txt_feat_len = datasets_mir.txt_feat_len

        self.CodeNet_s1_I = ShuffleNetV2(code_len=coed_length, net_size=0.5)
        self.CodeNet_s1_T = CNN_Text(code_len=coed_length, vocab_size = args.vocab_size, embedding_dim = args.embed_dim,
                                          filter_sizes = args.kernel_sizes, num_filters = args.kernel_num, l2_reg_lambda=0.0001)


        self.CodeNet_s2_I = ImgNet(code_len=coed_length)
        self.CodeNet_s2_T = CNN_Text(code_len=coed_length, vocab_size = args.vocab_size, embedding_dim = args.embed_dim,
                                          filter_sizes = args.kernel_sizes, num_filters = args.kernel_num, l2_reg_lambda=0.0001)

        self.Img_s1_ema = ShuffleNetV2(code_len=coed_length, net_size=0.5)
        self.Txt_s1_ema = CNN_Text(code_len=coed_length, vocab_size = args.vocab_size, embedding_dim = args.embed_dim,
                                          filter_sizes = args.kernel_sizes, num_filters = args.kernel_num, l2_reg_lambda=0.0001)

        self.Img_s2_ema = ImgNet(code_len=coed_length)
        self.Txt_s2_ema = CNN_Text(code_len=coed_length, vocab_size = args.vocab_size, embedding_dim = args.embed_dim,
                                          filter_sizes = args.kernel_sizes, num_filters = args.kernel_num, l2_reg_lambda=0.0001)


        self.opt_s1_I = torch.optim.SGD(self.CodeNet_s1_I.parameters(), lr=args.LR_IMG, momentum=args.MOMENTUM,weight_decay=args.WEIGHT_DECAY)
        self.opt_s1_T = torch.optim.SGD(self.CodeNet_s1_T.parameters(), lr=args.LR_TXT, momentum=args.MOMENTUM,weight_decay=args.WEIGHT_DECAY)
        self.opt_s2_I = torch.optim.SGD(self.CodeNet_s2_I.parameters(), lr=args.LR_IMG, momentum=args.MOMENTUM,weight_decay=args.WEIGHT_DECAY)
        self.opt_s2_T = torch.optim.SGD(self.CodeNet_s2_T.parameters(), lr=args.LR_TXT, momentum=args.MOMENTUM,weight_decay=args.WEIGHT_DECAY)

        self.global_step1 = 0
        self.global_step2 = 0

    def pre_train1(self, epoch, args):
        self.CodeNet_s1_I.cuda().train()
        self.CodeNet_s1_T.cuda().train()
        self.Img_s1_ema.cuda().train()
        self.Txt_s1_ema.cuda().train()
        self.FeatNet_I.cuda().eval()

        self.CodeNet_s1_I.set_alpha(epoch)
        self.CodeNet_s1_T.set_alpha(epoch)
        self.Img_s1_ema.set_alpha(epoch)
        self.Txt_s1_ema.set_alpha(epoch)

        logger.info('Epoch [%d/%d]' % (epoch + 1, args.NUM_EPOCH1))

        for idx, (img, F_T, txt, labels, _) in enumerate(self.train_loader):

            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            txt = Variable(torch.LongTensor(txt.numpy()).cuda())

            self.opt_s1_I.zero_grad()
            self.opt_s1_T.zero_grad()
            F_I, _, _ = self.FeatNet_I(img)

            # ------------------------------ student 1 ------------------------------------------------------
            batch_size_ = labels.size(0)
            noise_data1 = np.random.uniform(0, 1, 50 * args.embed_dim)
            noise_data1 = np.reshape(noise_data1, (50, args.embed_dim))
            te_noise1 = noise_data1[np.newaxis, :, :]
            te_noise_te1 = np.tile(te_noise1, (batch_size_, 1, 1))
            te_noise_tensor1 = torch.from_numpy(te_noise_te1).type(torch.FloatTensor)
            te_noise_tensor1 = Variable(te_noise_tensor1.cuda())

            Fea_I, _, code_I = self.CodeNet_s1_I(img)
            Fea_T, _, code_T = self.CodeNet_s1_T(txt, te_noise_tensor1)

            Dis_I = pairwise_distance(F.normalize(F_I), F.normalize(F_I))
            Dis_T = pairwise_distance(F.normalize(F_T), F.normalize(F_T))
            Stu_Dis_I = pairwise_distance(F.normalize(Fea_I), F.normalize(Fea_I))
            Stu_Dis_T = pairwise_distance(F.normalize(Fea_T), F.normalize(Fea_T))

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())

            F_I = F.normalize(F_I)
            F_T = F.normalize(F_T)
            S_I1 = F_I.mm(F_I.t())
            S_T1 = F_T.mm(F_T.t())
            S_I = S_I1 * 2 - 1
            S_T = S_T1 * 2 - 1

            S_tilde = args.BETA * S_I + (1 - args.BETA) * S_T
            # S = S_tilde * args.MU
            S = S_tilde
            Dis_S = args.BETA * Dis_I + (1 - args.BETA) * Dis_T

            Dis_S_Pos = S_tilde.detach().clone()
            Dis_S_Neg = S_tilde.detach().clone()

            Dis_S_Pos[Dis_S_Pos > 0] = 1
            Dis_S_Pos[Dis_S_Pos <= 0] = 0
            Dis_S_Neg[Dis_S_Neg > 0] = 0
            Dis_S_Neg[Dis_S_Neg < 0] = 1

            T_output_pos, T_output_neg = gaussian_kernel_matrix(Dis_S, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_I, output_neg_I = gaussian_kernel_matrix(Stu_Dis_I, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_T, output_neg_T = gaussian_kernel_matrix(Stu_Dis_T, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())

            teacher_output = torch.sigmoid((T_output_neg - T_output_pos) / 10)
            student_output_I = torch.sigmoid((output_neg_I - output_pos_I) / 10)
            student_output_T = torch.sigmoid((output_neg_T - output_pos_T) / 10)

            # calculate teacher_output_augmented
            teacher_output_augmented = torch.cat([teacher_output.unsqueeze(1), 1 - teacher_output.unsqueeze(1)],
                                                     dim=1)
            # calculate student_output_augmented
            student_output_augmented1 = torch.cat(
                    [student_output_I.unsqueeze(1), 1 - student_output_I.unsqueeze(1)], dim=1)
            student_output_augmented2 = torch.cat(
                    [student_output_T.unsqueeze(1), 1 - student_output_T.unsqueeze(1)], dim=1)

            loss4 = nn.KLDivLoss()(torch.log(student_output_augmented1), teacher_output_augmented) + nn.KLDivLoss()(
                    torch.log(student_output_augmented2), teacher_output_augmented)

            # ------------------------------ student 1 ------------------------------------------------------


            loss_s1_1 = F.mse_loss(BI_BI, S)
            loss_s1_2 = F.mse_loss(BI_BT, S)
            loss_s1_3 = F.mse_loss(BT_BT, S)
            loss_s1 = args.LAMBDA1 * loss_s1_1 + 1 * loss_s1_2 + args.LAMBDA2 * loss_s1_3 + 1000 * loss4

            loss_s1.backward()
            self.opt_s1_I.step()
            self.opt_s1_T.step()

            if (idx + 1) % (len(self.train_dataset) // args.batch_size / args.EPOCH_INTERVAL) == 0:
                logger.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                            % (epoch + 1, args.NUM_EPOCH1, idx + 1, len(self.train_dataset) // args.batch_size,
                                loss_s1.item()))

        if (epoch + 1) % (args.NUM_EPOCH1) == 0:
            self.global_step1 += 1
            update_ema_variables(self.CodeNet_s1_I, self.Img_s1_ema, args.ema_decay, self.global_step1)
            update_ema_variables(self.CodeNet_s1_T, self.Txt_s1_ema, args.ema_decay, self.global_step1)

    def pre_train2(self, epoch, args):
        self.CodeNet_s2_I.cuda().train()
        self.CodeNet_s2_T.cuda().train()
        self.Img_s2_ema.cuda().train()
        self.Txt_s2_ema.cuda().train()
        self.FeatNet_I.cuda().eval()

        self.CodeNet_s2_I.set_alpha(epoch)
        self.CodeNet_s2_T.set_alpha(epoch)
        self.Img_s2_ema.set_alpha(epoch)
        self.Txt_s2_ema.set_alpha(epoch)

        logger.info('Epoch [%d/%d]' % (epoch + 1, args.NUM_EPOCH1))

        for idx, (img, F_T, txt, labels, _) in enumerate(self.train_loader):

            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            txt = Variable(torch.LongTensor(txt.numpy()).cuda())

            self.opt_s2_I.zero_grad()
            self.opt_s2_T.zero_grad()
            F_I, _, _ = self.FeatNet_I(img)

            # ------------------------------ student 2 ------------------------------------------------------
            batch_size_ = labels.size(0)
            noise_data1 = np.random.uniform(0, 1, 50 * args.embed_dim)
            noise_data1 = np.reshape(noise_data1, (50, args.embed_dim))
            te_noise1 = noise_data1[np.newaxis, :, :]
            te_noise_te1 = np.tile(te_noise1, (batch_size_, 1, 1))
            te_noise_tensor1 = torch.from_numpy(te_noise_te1).type(torch.FloatTensor)
            te_noise_tensor1 = Variable(te_noise_tensor1.cuda())

            Fea_I, _, code_I = self.CodeNet_s2_I(img)
            Fea_T, _, code_T = self.CodeNet_s2_T(txt, te_noise_tensor1)

            Dis_I = pairwise_distance(F.normalize(F_I), F.normalize(F_I))
            Dis_T = pairwise_distance(F.normalize(F_T), F.normalize(F_T))
            Stu_Dis_I = pairwise_distance(F.normalize(Fea_I), F.normalize(Fea_I))
            Stu_Dis_T = pairwise_distance(F.normalize(Fea_T), F.normalize(Fea_T))

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())

            F_I = F.normalize(F_I)
            F_T = F.normalize(F_T)

            S_I1 = F_I.mm(F_I.t())
            S_T1 = F_T.mm(F_T.t())
            S_I = S_I1 * 2 - 1
            S_T = S_T1 * 2 - 1

            S = args.BETA * S_I + (1 - args.BETA) * S_T
            Dis_S = args.BETA * Dis_I + (1 - args.BETA) * Dis_T

            Dis_S_Pos = S.detach().clone()
            Dis_S_Neg = S.detach().clone()

            Dis_S_Pos[Dis_S_Pos > 0] = 1
            Dis_S_Pos[Dis_S_Pos <= 0] = 0
            Dis_S_Neg[Dis_S_Neg > 0] = 0
            Dis_S_Neg[Dis_S_Neg < 0] = 1

            T_output_pos, T_output_neg = gaussian_kernel_matrix(Dis_S, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_I, output_neg_I = gaussian_kernel_matrix(Stu_Dis_I, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_T, output_neg_T = gaussian_kernel_matrix(Stu_Dis_T, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())

            teacher_output = torch.sigmoid((T_output_neg - T_output_pos) / 10)
            student_output_I = torch.sigmoid((output_neg_I - output_pos_I) / 10)
            student_output_T = torch.sigmoid((output_neg_T - output_pos_T) / 10)

            # calculate teacher_output_augmented
            teacher_output_augmented = torch.cat([teacher_output.unsqueeze(1), 1 - teacher_output.unsqueeze(1)],
                                                 dim=1)
            # calculate student_output_augmented
            student_output_augmented1 = torch.cat(
                [student_output_I.unsqueeze(1), 1 - student_output_I.unsqueeze(1)], dim=1)
            student_output_augmented2 = torch.cat(
                [student_output_T.unsqueeze(1), 1 - student_output_T.unsqueeze(1)], dim=1)

            loss4 = nn.KLDivLoss()(torch.log(student_output_augmented1), teacher_output_augmented) + nn.KLDivLoss()(
                torch.log(student_output_augmented2), teacher_output_augmented)

            # ------------------------------ student 1 ------------------------------------------------------

            loss_s1_1 = F.mse_loss(BI_BI, S)
            loss_s1_2 = F.mse_loss(BI_BT, S)
            loss_s1_3 = F.mse_loss(BT_BT, S)
            loss_s1 = args.LAMBDA1 * loss_s1_1 + 1 * loss_s1_2 + args.LAMBDA2 * loss_s1_3 + 1000 * loss4

            loss_s1.backward()
            self.opt_s2_I.step()
            self.opt_s2_T.step()

            if (idx + 1) % (len(self.train_dataset) // args.batch_size / args.EPOCH_INTERVAL) == 0:
                logger.info('Epoch [%d/%d], Iter [%d/%d]  Loss2: %.4f '
                            % (epoch + 1, args.NUM_EPOCH1, idx + 1, len(self.train_dataset) // args.batch_size,
                               loss_s1.item()))

        if (epoch + 1) % (args.NUM_EPOCH1) == 0:
            self.global_step2 += 1
            update_ema_variables(self.CodeNet_s2_I, self.Img_s2_ema, args.ema_decay, self.global_step2)
            update_ema_variables(self.CodeNet_s2_T, self.Txt_s2_ema, args.ema_decay, self.global_step2)

    def post_train(self, epoch, args):
        self.CodeNet_s1_I.cuda().train()
        self.CodeNet_s1_T.cuda().train()
        self.CodeNet_s2_I.cuda().train()
        self.CodeNet_s2_T.cuda().train()

        self.FeatNet_I.cuda().eval()

        self.CodeNet_s1_I.set_alpha(epoch)
        self.CodeNet_s1_T.set_alpha(epoch)
        self.CodeNet_s2_I.set_alpha(epoch)
        self.CodeNet_s2_T.set_alpha(epoch)
        self.Img_s1_ema.set_alpha(epoch)
        self.Txt_s1_ema.set_alpha(epoch)
        self.Img_s2_ema.set_alpha(epoch)
        self.Txt_s2_ema.set_alpha(epoch)

        logger.info('Epoch [%d/%d]' % (epoch + 1, args.NUM_EPOCH2))
        for idx, (img, F_T, txt, labels, _) in enumerate(self.train_loader):

            #progressively increasing weights for similarity refinery
            p = (idx + epoch * len(self.train_dataset)) / (2 * (args.NUM_EPOCH2 * len(self.train_dataset)))

            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            txt = Variable(torch.LongTensor(txt.numpy()).cuda())

            self.opt_s1_I.zero_grad()
            self.opt_s1_T.zero_grad()
            self.opt_s2_I.zero_grad()
            self.opt_s2_T.zero_grad()

            batch_size_ = labels.size(0)

            center_crop = 224
            noise_data1 = np.random.uniform(0, 255, center_crop * center_crop * 3)
            im_noise1 = np.reshape(noise_data1, (3, center_crop, center_crop))
            im_noise1 = im_noise1[np.newaxis, :, :, :]
            im_noise_tr1 = np.tile(im_noise1, (batch_size_, 1, 1, 1))
            noise_tensor_tr1 = torch.from_numpy(im_noise_tr1).type(torch.FloatTensor)
            noise_tr1 = Variable(noise_tensor_tr1.cuda())
            img1 = img + noise_tr1 * 0.00001

            center_crop = 224
            noise_data2 = np.random.uniform(0, 255, center_crop * center_crop * 3)
            im_noise2 = np.reshape(noise_data2, (3, center_crop, center_crop))
            im_noise2 = im_noise2[np.newaxis, :, :, :]
            im_noise_tr2 = np.tile(im_noise2, (batch_size_, 1, 1, 1))
            noise_tensor_tr2 = torch.from_numpy(im_noise_tr2).type(torch.FloatTensor)
            noise_tr2 = Variable(noise_tensor_tr2.cuda())
            img2 = img + noise_tr2 * 0.00001

            noise_data1 = np.random.uniform(0, 1, 50 * args.embed_dim)
            noise_data1 = np.reshape(noise_data1, (50, args.embed_dim))
            te_noise1 = noise_data1[np.newaxis, :, :]
            te_noise_te1 = np.tile(te_noise1, (batch_size_, 1, 1))
            te_noise_tensor1 = torch.from_numpy(te_noise_te1).type(torch.FloatTensor)
            te_noise_tensor1 = Variable(te_noise_tensor1.cuda())

            noise_data2 = np.random.uniform(0, 1, 50 * args.embed_dim)
            noise_data2 = np.reshape(noise_data2, (50, args.embed_dim))
            te_noise2 = noise_data2[np.newaxis, :, :]
            te_noise_te2 = np.tile(te_noise2, (batch_size_, 1, 1))
            te_noise_tensor2 = torch.from_numpy(te_noise_te2).type(torch.FloatTensor)
            te_noise_tensor2 = Variable(te_noise_tensor2.cuda())

            Fea_s1_I, _, code_s1_I = self.CodeNet_s1_I(img1)
            Fea_s1_T, _, code_s1_T = self.CodeNet_s1_T(txt,te_noise_tensor1)
            Fea_s2_I, _, code_s2_I = self.CodeNet_s2_I(img2)
            Fea_s2_T, _, code_s2_T = self.CodeNet_s2_T(txt,te_noise_tensor2)

            with torch.no_grad():
                ema_img1 = Variable(img1.cuda())
                ema_img2 = Variable(img2.cuda())
                ema_txt = txt
                Fea_s1_ema_I, _, ema_s1_I = self.Img_s1_ema(ema_img1)
                Fea_s1_ema_T, _, ema_s1_T = self.Txt_s1_ema(ema_txt,te_noise_tensor1)
                Fea_s2_ema_I, _, ema_s2_I = self.Img_s2_ema(ema_img2)
                Fea_s2_ema_T, _, ema_s2_T = self.Txt_s2_ema(ema_txt,te_noise_tensor2)

            # --------------- cross-task teacher --------------------------
            F_I, _, _ = self.FeatNet_I(img)
            F_I = F.normalize(F_I)
            F_T = F.normalize(F_T)
            S_I = F_I.mm(F_I.t())
            S_T = F_T.mm(F_T.t())
            S_I = S_I * 2 - 1
            S_T = S_T * 2 - 1
            S_cross = args.BETA * S_I + (1 - args.BETA) * S_T

            Dis_I = pairwise_distance(F_I, F_I)
            Dis_T = pairwise_distance(F_T, F_T)
            Dis_S = args.BETA * Dis_I + (1 - args.BETA) * Dis_T

            # ------------- mean teacher --------------------------------
            A_I_s1 = F.normalize(ema_s1_I)
            A_T_s1 = F.normalize(ema_s1_T)
            A_I_s2 = F.normalize(ema_s2_I)
            A_T_s2 = F.normalize(ema_s2_T)
            S_mean1 =  (A_I_s1.mm(A_T_s1.t()) + A_T_s1.mm(A_I_s1.t()))/2
            S_mean2 =  (A_I_s2.mm(A_T_s2.t()) + A_T_s2.mm(A_I_s2.t()))/2

            # ------------------------------ embedding ------------------------------------------------------
            B_I_s1 = F.normalize(code_s1_I)
            B_T_s1 = F.normalize(code_s1_T)
            B_I_s2 = F.normalize(code_s2_I)
            B_T_s2 = F.normalize(code_s2_T)

            # ------------------------------------------------------------------------------------------------
            Stu_Dis_I_ema_s1 = pairwise_distance(F.normalize(Fea_s1_ema_I), F.normalize(Fea_s1_ema_I))
            Stu_Dis_T_ema_s1 = pairwise_distance(F.normalize(Fea_s1_ema_T), F.normalize(Fea_s1_ema_T))
            Stu_Dis_I_ema_s2 = pairwise_distance(F.normalize(Fea_s2_ema_I), F.normalize(Fea_s2_ema_I))
            Stu_Dis_T_ema_s2 = pairwise_distance(F.normalize(Fea_s2_ema_T), F.normalize(Fea_s2_ema_T))

            Stu_Dis_I_s1 = pairwise_distance(F.normalize(Fea_s1_I), F.normalize(Fea_s1_I))
            Stu_Dis_T_s1 = pairwise_distance(F.normalize(Fea_s1_T), F.normalize(Fea_s1_T))
            Stu_Dis_I_s2 = pairwise_distance(F.normalize(Fea_s2_I), F.normalize(Fea_s2_I))
            Stu_Dis_T_s2 = pairwise_distance(F.normalize(Fea_s2_T), F.normalize(Fea_s2_T))
            Dis_S_s1 = (Stu_Dis_I_ema_s1 + Stu_Dis_T_ema_s1) / 2
            Dis_S_s2 = (Stu_Dis_I_ema_s2 + Stu_Dis_T_ema_s2) / 2

            S = ((1 - p) * S_cross + (p) * ((1 - args.alpha) * S_mean1 + (args.alpha) * S_mean2))
            Dis_S = ((1 - p) * Dis_S + (p) * ((1 - args.alpha) * Dis_S_s1 + (args.alpha) * Dis_S_s2))

            Dis_S_Pos = S.detach().clone()
            Dis_S_Neg = S.detach().clone()

            Dis_S_Pos[Dis_S_Pos > 0] = 1
            Dis_S_Pos[Dis_S_Pos <= 0] = 0
            Dis_S_Neg[Dis_S_Neg > 0] = 0
            Dis_S_Neg[Dis_S_Neg < 0] = 1

            T_output_pos, T_output_neg = gaussian_kernel_matrix(Dis_S, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_I_s1, output_neg_I_s1 = gaussian_kernel_matrix(Stu_Dis_I_s1, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_T_s1, output_neg_T_s1 = gaussian_kernel_matrix(Stu_Dis_T_s1, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_I_s2, output_neg_I_s2 = gaussian_kernel_matrix(Stu_Dis_I_s2, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())
            output_pos_T_s2, output_neg_T_s2 = gaussian_kernel_matrix(Stu_Dis_T_s2, Dis_S_Pos.cuda(), Dis_S_Neg.cuda())

            teacher_output = torch.sigmoid((T_output_neg - T_output_pos) / 10)
            student_output_I_s1 = torch.sigmoid((output_neg_I_s1 - output_pos_I_s1) / 10)
            student_output_T_s1 = torch.sigmoid((output_neg_T_s1 - output_pos_T_s1) / 10)
            student_output_I_s2 = torch.sigmoid((output_neg_I_s2 - output_pos_I_s2) / 10)
            student_output_T_s2 = torch.sigmoid((output_neg_T_s2 - output_pos_T_s2) / 10)

            # calculate teacher_output_augmented
            teacher_output_augmented = torch.cat([teacher_output.unsqueeze(1), 1 - teacher_output.unsqueeze(1)],
                                                 dim=1)
            # calculate student_output_augmented
            student_output_augmented1_s1 = torch.cat(
                [student_output_I_s1.unsqueeze(1), 1 - student_output_I_s1.unsqueeze(1)], dim=1)
            student_output_augmented2_s1 = torch.cat(
                [student_output_T_s1.unsqueeze(1), 1 - student_output_T_s1.unsqueeze(1)], dim=1)
            student_output_augmented1_s2 = torch.cat(
                [student_output_I_s2.unsqueeze(1), 1 - student_output_I_s2.unsqueeze(1)], dim=1)
            student_output_augmented2_s2 = torch.cat(
                [student_output_T_s2.unsqueeze(1), 1 - student_output_T_s2.unsqueeze(1)], dim=1)

            loss4 = nn.KLDivLoss()(torch.log(student_output_augmented1_s1), teacher_output_augmented) + nn.KLDivLoss()(
                torch.log(student_output_augmented2_s1), teacher_output_augmented)
            loss5 = nn.KLDivLoss()(torch.log(student_output_augmented1_s2), teacher_output_augmented) + nn.KLDivLoss()(
                torch.log(student_output_augmented2_s2), teacher_output_augmented)

            # ------------------------------ student 1 optimization ------------------------------------------------------
            BI_BI_s1 = B_I_s1.mm(B_I_s1.t())
            BT_BT_s1 = B_T_s1.mm(B_T_s1.t())
            BI_BT_s1 = B_I_s1.mm(B_T_s1.t())

            loss1 = F.mse_loss(BI_BI_s1, S)
            loss2 = F.mse_loss(BI_BT_s1, S)
            loss3 = F.mse_loss(BT_BT_s1, S)
            loss_s1 = args.LAMBDA1 * loss1 + 1 * loss2 + args.LAMBDA2 * loss3

            # ------------------------------ student 2 optimization ------------------------------------------------------

            BI_BI_s2 = B_I_s2.mm(B_I_s2.t())
            BT_BT_s2 = B_T_s2.mm(B_T_s2.t())
            BI_BT_s2 = B_I_s2.mm(B_T_s2.t())
            loss1 = F.mse_loss(BI_BI_s2, S)
            loss2 = F.mse_loss(BI_BT_s2, S)
            loss3 = F.mse_loss(BT_BT_s2, S)

            loss_s2 = args.LAMBDA1 * loss1 + 1 * loss2 + args.LAMBDA2 * loss3 + 1000 * (loss4 + loss5)

            loss = loss_s1 + loss_s2

            loss.backward()

            self.opt_s1_I.step()
            self.opt_s1_T.step()
            self.opt_s2_I.step()
            self.opt_s2_T.step()

            self.global_step1 += 1
            self.global_step2 += 1
            update_ema_variables(self.CodeNet_s1_I, self.Img_s1_ema, args.ema_decay, self.global_step1)
            update_ema_variables(self.CodeNet_s1_T, self.Txt_s1_ema, args.ema_decay, self.global_step1)
            update_ema_variables(self.CodeNet_s2_I, self.Img_s2_ema, args.ema_decay, self.global_step2)
            update_ema_variables(self.CodeNet_s2_T, self.Txt_s2_ema, args.ema_decay, self.global_step2)

            if (idx + 1) % (len(self.train_dataset) // args.batch_size / args.EPOCH_INTERVAL) == 0:
                logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f'
                            % (epoch + 1, args.NUM_EPOCH2, idx + 1, len(self.train_dataset) // args.batch_size,
                               loss_s1.item(), loss_s2.item()))

    def eval(self, epoch, bit):
        logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_s1_I.eval().cuda()
        self.CodeNet_s1_T.eval().cuda()
        self.CodeNet_s2_I.eval().cuda()
        self.CodeNet_s2_T.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_s1_I,
                                                        self.CodeNet_s1_T, self.database_dataset, self.test_dataset,args)
        re_BI_s2, re_BT_s2, re_L_s2, qu_BI_s2, qu_BT_s2, qu_L_s2 = compress(self.database_loader, self.test_loader, self.CodeNet_s2_I,
                                                          self.CodeNet_s2_T, self.database_dataset, self.test_dataset,args)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_I2T1 = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=500)
        MAP_T2I1 = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=500)
        MAP_I2T_s2 = calculate_top_map(qu_B=qu_BI_s2, re_B=re_BT_s2, qu_L=qu_L_s2, re_L=re_L_s2, topk=50)
        MAP_T2I_s2 = calculate_top_map(qu_B=qu_BT_s2, re_B=re_BI_s2, qu_L=qu_L_s2, re_L=re_L_s2, topk=50)

        if (self.best_it + self.best_ti) < (MAP_I2T1 + MAP_T2I1):
            self.best_it = MAP_I2T1
            self.best_ti = MAP_T2I1

        sio.savemat('./result/' + str(bit) + '/' + str(epoch) + '.mat', {
                            're_BI': re_BI,
                            're_BT': re_BT,
                            're_L': re_L,
                            'qu_BI': qu_BI,
                            'qu_BT': qu_BT,
                            'qu_L': qu_L})

        logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        logger.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (self.best_it, self.best_ti))
        logger.info('MAP_s2 of Image to Text: %.3f, MAP_s2 of Text to Image: %.3f' % (MAP_I2T_s2, MAP_T2I_s2))
        logger.info('MAP 500 of Image to Text: %.3f, MAP 500 of Text to Image: %.3f' % (MAP_I2T1, MAP_T2I1))
        logger.info('--------------------------------------------------------------------')

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def gaussian_kernel_matrix(dist, S1, S2):
    k1 = 1 / (torch.sum(S1, 1) + 0.01)
    k2 = 1 / (torch.sum(S2, 1) + 0.01)

    dist1 = dist * S1
    dist2 = dist * S2
    dist1 = torch.sum(dist1, 0)
    dist2 = torch.sum(dist2, 0)

    return dist1 * k1, dist2 * k2


def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.norm(x - y, p=2, dim=1)
    output = torch.transpose(output, 0, 1)

    return output


def mkdir_multi(path):
    # Check whether the path exists
    isExists = os.path.exists(path)

    if not isExists:
        # If it does not, create a directory (multilayer).
        os.makedirs(path)
        print('successfully creat path！')
        return True
    else:
        # If the directory exists, the system does not create the directory
        # and displays a message indicating that the directory already exists
        print('path already exists！')
        return False


def _logging():
    global logger
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
        logdir = './result/' + str(bit) + '/'
        mkdir_multi(logdir)
        _logging()

        if args.EVAL == True:
            sess.load_checkpoints()
        else:
            logger.info('--------------------------pre_train Stage--------------------------')
            sess.define_model(bit)

            for epoch in range(args.NUM_EPOCH1):
                # warmup the Model
                iter_time1 = time.time()
                sess.pre_train2(epoch, args)
                iter_time1 = time.time() - iter_time1
                logger.info('[pre_train time: %.4f]', iter_time1)
                if (epoch + 1) % args.EVAL_INTERVAL == 0:
                    iter_time1_1 = time.time()
                    sess.eval(epoch, bit)
                    iter_time1_1 = time.time() - iter_time1_1
                    logger.info('[pre_train eval2 time: %.4f]', iter_time1_1)

            for epoch in range(args.NUM_EPOCH1):
                # warmup the Model
                iter_time0 = time.time()
                sess.pre_train1(epoch, args)
                iter_time0 = time.time() - iter_time0
                logger.info('[pre_train time: %.4f]', iter_time0)
                if (epoch + 1) % args.EVAL_INTERVAL == 0:
                    iter_time0_1 = time.time()
                    sess.eval(epoch, bit)
                    iter_time0_1 = time.time() - iter_time0_1
                    logger.info('[pre_train1 eval time: %.4f]', iter_time0_1)


            logger.info('--------------------------train Stage--------------------------')
            for epoch in range(args.NUM_EPOCH2):
                # train the Model
                iter_time2 = time.time()
                sess.post_train(epoch, args)
                iter_time2 = time.time() - iter_time2
                logger.info('[train time: %.4f]', iter_time2)
                if (epoch + 1) % args.EVAL_INTERVAL == 0:
                    iter_time3 = time.time()
                    sess.eval(epoch, bit)
                    iter_time3 = time.time() - iter_time3
                    logger.info('[train eval time: %.4f]', iter_time3)


if __name__=="__main__":
    main()