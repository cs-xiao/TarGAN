from models import Generator,Discriminator,ShapeUNet
from data_loader import *
from torch.utils.data import DataLoader
from utils import *
import argparse
import time
import matplotlib.pyplot as plt
import datetime
from visualdl import LogWriter
import copy
import torch.nn.functional as F

def train(args):
    glr = args.lr
    dlr = args.ttur
    print(glr, dlr)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    set_seed(args.random_seed)

    syn_dataset = ChaosDataset_Syn_new(split='train', modals=args.modals)
    syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True)
    syneval_dataset = ChaosDataset_Syn_Test(modal=args.modals[0], gan=True)
    syneval_dataset2 = ChaosDataset_Syn_Test(modal=args.modals[1], gan=True)
    syneval_dataset3 = ChaosDataset_Syn_Test(modal=args.modals[2], gan=True)
    netG = Generator(1 + args.c_dim, args.G_conv, 2, 3, True, True)
    netH = ShapeUNet(img_ch=1, mid=args.h_conv)
    netD_i = Discriminator(c_dim=args.c_dim * 2, image_size=256)
    netD_t = Discriminator(c_dim=args.c_dim * 2, image_size=256)

    g_optimizier = torch.optim.Adam(netG.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))
    di_optimizier = torch.optim.Adam(netD_i.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    dt_optimizier = torch.optim.Adam(netD_t.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    h_optimizier = torch.optim.Adam(netH.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))

    netG_use = copy.deepcopy(netG)
    netG.to(device)
    netD_i.to(device)
    netD_t.to(device)
    netH.to(device)
    netG_use.to(device)

    start_time = time.time()
    print('start training...')

    ii = 0
    logdir = "../log/" + args.save_path
    log_writer = LogWriter(logdir)
    for i in range(args.sepoch,args.epoch):
        for epoch, (x_real, t_img, shape_mask, mask, label_org) in enumerate(syn_loader):
            # 1. Preprocess input data
            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            c_org = label2onehot(label_org, args.c_dim)
            c_trg = label2onehot(label_trg, args.c_dim)
            d_false_org = label2onehot(label_org + args.c_dim, args.c_dim * 2)
            d_org = label2onehot(label_org, args.c_dim * 2)
            g_trg = label2onehot(label_trg, args.c_dim * 2)
            x_real = x_real.to(device)  # Input images.
            c_org = c_org.to(device)  # Original domain labels.
            c_trg = c_trg.to(device)  # Target domain labels.
            d_org = d_org.to(device)  # Labels for computing classification loss.
            g_trg = g_trg.to(device)  # Labels for computing classification loss.
            d_false_org = d_false_org.to(device) # Labels for computing classification loss.
            mask = mask.to(device)
            shape_mask = shape_mask.to(device)
            t_img = t_img.to(device)
            index = loss_filter(mask)
            # 2. Train the discriminator
            # Compute loss with real whole images.
            out_src, out_cls = netD_i(x_real)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, d_org, reduction='sum') / out_cls.size(0)

            # Compute loss with fake whole images.
            with torch.no_grad():
                x_fake, t_fake = netG(x_real, t_img, c_trg)
            out_src, out_f_cls = netD_i(x_fake.detach())
            d_loss_fake = torch.mean(out_src)
            d_loss_f_cls = F.binary_cross_entropy_with_logits(out_f_cls, d_false_org, reduction='sum') / out_f_cls.size(0)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = netD_i(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat, device)

            # compute loss with target images
            if index.shape[0] != 0:
                out_src, out_cls = netD_t(torch.index_select(t_img, dim=0, index=index))
                d_org = torch.index_select(d_org, dim=0, index=index)
                d_loss_real_t = -torch.mean(out_src)
                d_loss_cls_t = F.binary_cross_entropy_with_logits(out_cls, d_org, reduction='sum') / out_cls.size(0)

                out_src, out_f_cls = netD_t(torch.index_select(t_fake.detach(), dim=0, index=index))
                d_false_org = torch.index_select(d_false_org, dim=0, index=index)
                d_loss_fake_t = torch.mean(out_src)
                d_loss_f_cls_t = F.binary_cross_entropy_with_logits(out_f_cls, d_false_org,
                                                                    reduction='sum') / out_f_cls.size(0)

                x_hat = (alpha * t_img.data + (1 - alpha) * t_fake.data).requires_grad_(True)
                x_hat = torch.index_select(x_hat, dim=0, index=index)
                out_src, _ = netD_t(x_hat)
                d_loss_gp_t = gradient_penalty(out_src, x_hat, device)

                dt_loss = d_loss_real_t + d_loss_fake_t + d_loss_cls_t + d_loss_gp_t * 10 + d_loss_f_cls_t * args.w_d_false_t_c
                w_dt = (-d_loss_real_t - d_loss_fake_t).item()
            else:
                dt_loss = torch.FloatTensor([0]).to(device)
                w_dt = 0
                d_loss_f_cls_t = torch.FloatTensor([0]).to(device)
            # Backward and optimize.
            di_loss = d_loss_real + d_loss_fake + d_loss_cls + d_loss_gp * 10 + d_loss_f_cls * args.w_d_false_c
            d_loss = di_loss + dt_loss
            w_di = (-d_loss_real - d_loss_fake).item()

            g_optimizier.zero_grad()
            di_optimizier.zero_grad()
            dt_optimizier.zero_grad()
            d_loss.backward()
            di_optimizier.step()
            dt_optimizier.step()

            #  3. Train the generator
            # Original-to-target domain.
            x_fake, t_fake = netG(x_real, t_img, c_trg)
            out_src, out_cls = netD_i(x_fake)
            g_loss_fake = -torch.mean(out_src)
            g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, g_trg, reduction='sum') / out_cls.size(0)

            shape_loss = F.mse_loss(netH(x_fake), shape_mask.float())
            # Target-to-original domain.
            x_reconst, t_reconst = netG(x_fake, t_fake, c_org)
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

            if index.shape[0] != 0:
                out_src, out_cls = netD_t(torch.index_select(t_fake, dim=0, index=index))
                g_trg = torch.index_select(g_trg, dim=0, index=index)
                g_loss_fake_t = -torch.mean(out_src)
                g_loss_cls_t = F.binary_cross_entropy_with_logits(out_cls, g_trg, reduction='sum') / out_cls.size(0)
                gt_loss = g_loss_fake_t + g_loss_cls_t * args.w_g_t_c
            else:
                gt_loss = torch.FloatTensor([0]).to(device)
                g_loss_cls_t = torch.FloatTensor([0]).to(device)

            shape_loss_t = F.mse_loss(netH(t_fake), mask.float())
            g_loss_rec_t = torch.mean(torch.abs(t_img - t_reconst))
            cross_loss = torch.mean(torch.abs(denorm(x_fake) * mask - denorm(t_fake)))
            # Backward and optimize.
            gi_loss = g_loss_fake + args.w_cycle * g_loss_rec + g_loss_cls * args.w_g_c + shape_loss* args.w_shape
            gt_loss = gt_loss + args.w_cycle * g_loss_rec_t + shape_loss_t* args.w_shape + cross_loss * args.w_g_cross
            g_loss = gi_loss + gt_loss

            g_optimizier.zero_grad()
            di_optimizier.zero_grad()
            dt_optimizier.zero_grad()
            h_optimizier.zero_grad()
            g_loss.backward()
            g_optimizier.step()
            h_optimizier.step()

            moving_average(netG, netG_use, beta=0.999)

            if (epoch + 0) % 10 == 0:
                log_writer.add_scalar(tag="train/D/w_di", step=ii, value=w_di)
                log_writer.add_scalar(tag="train/D/w_dt", step=ii, value=w_dt)
                log_writer.add_scalar(tag="train/D/loss_f_cls", step=ii, value=d_loss_f_cls.item())
                log_writer.add_scalar(tag="train/D/loss_f_cls_t", step=ii, value=d_loss_f_cls_t.item())
                log_writer.add_scalar(tag="train/G/loss_cls", step=ii, value=g_loss_cls.item())
                log_writer.add_scalar(tag="train/G/loss_cls_t", step=ii, value=g_loss_cls_t.item())
                log_writer.add_scalar(tag="train/G/loss_shape", step=ii, value=shape_loss.item())
                log_writer.add_scalar(tag="train/G/loss_shape_t", step=ii, value=shape_loss_t.item())
                log_writer.add_scalar(tag="train/G/loss_cross", step=ii, value=cross_loss.item())

            ii = ii + 1
            ###################################

        if (i + 1) % 1 == 0 and (i + 1) > 0:
            # show syn images after every epoch
            plt.figure(dpi=120)
            with torch.no_grad():
                index = random.choice([0,1,2])
                if index == 0:
                    img = syneval_dataset[13][0]
                    mask = syneval_dataset[13][1].to(device)
                elif index == 1:
                    img = syneval_dataset2[3][0]
                    mask = syneval_dataset2[3][1].to(device)
                else:
                    img = syneval_dataset3[43][0]
                    mask = syneval_dataset3[43][1].to(device)
                img = img.unsqueeze(dim=0).to(device)
                pred_t1_img = netG_use(img, None, c=getLabel(img, device, 0, args.c_dim), mode='test')
                pred_t2_img = netG_use(img, None, c=getLabel(img, device, 1, args.c_dim), mode='test')
                pred_t3_img = netG_use(img, None, c=getLabel(img, device, 2, args.c_dim), mode='test')
                plt.subplot(221)
                plt.imshow(denorm(img).squeeze().cpu().numpy(), cmap='gray')
                plt.title(str(i + 1) + '_source')
                plt.subplot(242)
                plt.imshow(denorm(pred_t1_img).squeeze().cpu().numpy(), cmap='gray')
                plt.title('pred_x1')
                plt.subplot(243)
                plt.imshow(denorm(pred_t2_img).squeeze().cpu().numpy(), cmap='gray')
                plt.title('pred_x2')
                plt.subplot(244)
                plt.imshow(denorm(pred_t3_img).squeeze().cpu().numpy(), cmap='gray')
                plt.title('pred_x3')
                plt.show()

        if (i + 1) == args.epoch:
            args.net_name = 'netG'
            save_state_net(netG, args, i + 1, None)
            args.net_name = 'netG_use'
            save_state_net(netG_use, args, i + 1, None)
            args.net_name = 'netDi'
            save_state_net(netD_i, args, i + 1, None)
            args.net_name = 'netDt'
            save_state_net(netD_t, args, i + 1, None)

        if (i + 1) % 1 == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.epoch)
            print(log)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datasets', type=str, default='chaos')
    parser.add_argument('-save_path', type=str, default='s3m-gan-chaos-v1.0')
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-gan_version', type=str, default='Generator[2/3]+shapeunet+D')
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-sepoch', type=int, default=0)
    parser.add_argument('-modals', type=tuple, default=('t1', 't2', 'ct'))
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-loss_function', type=str, default='wgan-gp+move+cycle+ugan+d+l2')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-note', type=str,default='affine:True;')
    parser.add_argument('-random_seed', type=int, default='888')
    parser.add_argument('-c_dim', type=int, default='3')
    parser.add_argument('-h_conv', type=int, default='16')
    parser.add_argument('-G_conv', type=int, default='64')
    parser.add_argument('-betas', type=tuple, default=(0.5, 0.9))
    parser.add_argument('-ttur', type=float, default=3e-4)
    parser.add_argument('-w_d_false_c', type=float, default=0.01)
    parser.add_argument('-w_d_false_t_c', type=float, default=0.01)
    parser.add_argument('-w_g_c', type=float, default=1.0)
    parser.add_argument('-w_g_t_c', type=float, default=1.0)
    parser.add_argument('-w_g_cross', type=float, default=50.0)
    parser.add_argument('-w_shape', type=float, default=1)
    parser.add_argument('-w_cycle', type=float, default=1)
    args = parser.parse_args()
    print(args)
    train(args)
