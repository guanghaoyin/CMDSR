from options.option_Train_CMDSR import set_args
from tools.utils import check_args
from data_loader import SRMultiGroupRandomTaskDataSet, SRDataSet
from models.CMDSR import CMDSR
from losses.Task_Contrastive_Loss import TaskContrastiveLoss
import math
import os
import torch
from torch.utils.data import DataLoader
from PSNR_SSIM_cal import cal_PSNR, cal_SSIM
import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return np.uint8((img*pixel_range).round())

def cal_params(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("Total parameters is :" + str(k))

def main():
    #Get args
    args = set_args()
    args = check_args(args)
    
    #to device finishing during model initial
    device = torch.device('cuda' if args.gpu_ids is not None else 'cpu')
    #model
    metaNet = CMDSR(args=args)
    if args.load_trained_model:
        print('loading trained model {}'.format(args.load_trained_model_path))
        metaNet._load_pretrain_net()
    metaNet = metaNet.to(device)
    cal_params(metaNet)
    #dataset
    args.support_size = args.support_size*2 
    train_dataset = SRMultiGroupRandomTaskDataSet(args=args, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.task_size, num_workers=1, shuffle=True)
    args.support_size = args.support_size//2
    
    test_dataset = SRDataSet(args=args, is_train=False, is_meta=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    #loss
    train_pixel_loss = torch.nn.L1Loss()
    #optimizer
    if args.optmize_sr_condition_net:
        train_optimizer = torch.optim.Adam(metaNet.parameters(), lr = args.lr)
    else:
        train_optimizer = torch.optim.Adam(metaNet.sr_net.parameters(), lr = args.lr)

    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(train_optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    
    if args.use_support_Mod and args.opt_condition_net_sep:
        Contrastive_loss = TaskContrastiveLoss()
        conditionnet_mix_optimizer =  torch.optim.Adam(metaNet.condition_net.parameters(), lr = args.lr_condition)
        conditionnet_mix_scheduler = torch.optim.lr_scheduler.MultiStepLR(conditionnet_mix_optimizer, milestones=args.milestones, gamma=args.lr_gamma_condition)

    #size and length cal
    train_support_size = int(math.ceil(train_dataset.dataset_len / (args.task_size*args.support_size)))
    total_iters = int(args.niter)
    total_support_epoches = int(math.ceil(total_iters / train_support_size))
    current_step = 0

    #run/time
    print('The batch nums of training support loader {:,d}'.format(len(train_loader)))
    print('Total support epoch {}'.format(total_support_epoches))

    train_step_visual_list = []
    train_losses_visual_list = []
    train_conditionNet_step_visual_list = []
    train_conditinNet_losses_visual_list = []
    train_inner_class_distance_visual_list = []
    train_cross_class_distance_visual_list = []
    test_step_visual_list = []
    test_psnr_visual_list = []
    test_ssim_visual_list = []

    txt_psnr_path = os.path.join(args.log_checkpoint_dir,'txt_psnr.txt')
    txt_ssim_path = os.path.join(args.log_checkpoint_dir,'txt_ssim.txt')
    txt_loss_path = os.path.join(args.log_checkpoint_dir,'txt_loss.txt')

    for epoch in range(total_support_epoches):
        print('Epoch : {}'.format(epoch))
        metaNet.train()
        for lr_double_patchs, hr_double_patchs, names, deg_cof in train_loader:
        #[task_size, support_size*2, 3, h, w]
            if current_step%args.run_test_every == 0:
                # TODO eval
                psnr_cur_img_list = []
                ssim_cur_img_list = []

            #Get degradated-lr and HR
            task_size = int(lr_double_patchs.shape[0])
            support_size = int(lr_double_patchs.shape[1]//2)
            
            # optimize condition net
            if current_step%args.opt_condition_net_step == 0:
                #cal_contrastive_loss
                half_support_x1 = deepcopy(lr_double_patchs[:,0:lr_double_patchs.shape[1]//2,:,:,:])
                half_support_x2 = deepcopy(lr_double_patchs[:,lr_double_patchs.shape[1]//2:,:,:,:])
                half_support_x1 = half_support_x1.reshape((task_size,-1,half_support_x1.shape[3],half_support_x1.shape[4])).to(device)
                half_support_x2 = half_support_x2.reshape((task_size,-1,half_support_x2.shape[3],half_support_x2.shape[4])).to(device)
                
                conditionnet_mix_optimizer.zero_grad()

                half_condition_feature1 = metaNet.condition_net(half_support_x1)
                half_condition_feature2 = metaNet.condition_net(half_support_x2)
                
                contrastive_loss, inner_class_distance, cross_class_distance = Contrastive_loss(half_condition_feature1, half_condition_feature2)
                
                #cal_sr_loss
                hr_support_x1 = deepcopy(hr_double_patchs[:,0:hr_double_patchs.shape[1]//2,:,:,:]).reshape((-1,3,hr_double_patchs.shape[3],hr_double_patchs.shape[4])).to(device)
                lr_support_x1 = deepcopy(half_support_x1).reshape((-1,3,half_support_x1.shape[2],half_support_x1.shape[3])).to(device)
                sr_support_x1 = metaNet(x=lr_support_x1, support_x=half_support_x1)
                sr_support_loss = train_pixel_loss(sr_support_x1, hr_support_x1)

                
                condition_loss = contrastive_loss + sr_support_loss*args.coefficient_contrastive_l1
                if current_step % args.print_train_loss_every == 0:
                    print('Meta step:{} conditinNet loss:{} '.format(current_step, condition_loss.cpu().detach().item()))
                    print('Meta step:{} inner_class distance:{} '.format(current_step, inner_class_distance))
                    print('Meta step:{} cross_class distance:{} '.format(current_step, cross_class_distance))
                    train_conditionNet_step_visual_list.append(current_step)
                    train_conditinNet_losses_visual_list.append(condition_loss.cpu().detach().item())
                    train_inner_class_distance_visual_list.append(inner_class_distance)
                    train_cross_class_distance_visual_list.append(cross_class_distance)
                    if current_step % args.save_train_loss_every == 0:
                        plt.xlabel('Iteration')
                        plt.ylabel('ConditinNet loss')
                        plt.plot(np.array(train_conditionNet_step_visual_list), np.array(train_conditinNet_losses_visual_list), label='ConditinNet loss')
                        plt.legend()
                        plt.title('ConditinNet loss')
                        plt.savefig(os.path.join(args.log_checkpoint_dir, 'Meta_ResNet_Mod_ConditinNet_losses.png'))
                        plt.close('all')
                        
                        plt.xlabel('Iteration')
                        plt.ylabel('class distance')
                        plt.plot(np.array(train_conditionNet_step_visual_list), np.array(train_inner_class_distance_visual_list), color='b', label='inner_class distance')
                        plt.plot(np.array(train_conditionNet_step_visual_list), np.array(train_cross_class_distance_visual_list), color='r', label='cross_class distance')
                        plt.legend()
                        plt.title('class distance')
                        plt.savefig(os.path.join(args.log_checkpoint_dir, 'Meta_ResNet_Mod_class_distance.png'))
                        plt.close('all')

                condition_loss.backward()
                conditionnet_mix_optimizer.step()
                conditionnet_mix_scheduler.step()

                torch.cuda.empty_cache()

            for i in range(2):
                lr_patchs = lr_double_patchs[:,support_size*i:support_size*(i+1),:,:,:]
                hr_patchs = hr_double_patchs[:,support_size*i:support_size*(i+1),:,:,:]
                lr_support_patchs = deepcopy(lr_patchs).reshape((task_size,-1,lr_patchs.shape[3],lr_patchs.shape[4])).to(device)
                lr_patchs = lr_patchs.reshape((task_size*support_size,lr_patchs.shape[2],lr_patchs.shape[3],lr_patchs.shape[4])).to(device)
                hr_patchs = hr_patchs.reshape((task_size*support_size,hr_patchs.shape[2],hr_patchs.shape[3],hr_patchs.shape[4])).to(device)
                
                train_optimizer.zero_grad()
                
                sr_support_patchs = metaNet(x=lr_patchs, support_x=lr_support_patchs)
                sr_l1_loss = train_pixel_loss(sr_support_patchs, hr_patchs)

                if current_step % args.print_train_loss_every == 0:
                    print('Meta step:{} loss:{} '.format(current_step, sr_l1_loss.cpu().detach().item()))
                    train_step_visual_list.append(current_step)
                    train_losses_visual_list.append(sr_l1_loss.cpu().detach().item())
                    np.savetxt(txt_loss_path,np.array(train_losses_visual_list))

                if current_step % args.save_train_loss_every == 0:
                    plt.xlabel('Iteration')
                    plt.ylabel('L1 loss')
                    plt.plot(np.array(train_step_visual_list), np.array(train_losses_visual_list), label='training losses')
                    plt.legend()
                    plt.title('training losses')
                    plt.savefig(os.path.join(args.log_checkpoint_dir, 'Meta_ResNet_Mod_training_losses.png'))
                    plt.close('all')

                sr_l1_loss.backward()
                train_optimizer.step()
                train_scheduler.step()
                
                if current_step%args.run_test_every == 0:
                    psnr_cur_img_list = []
                    ssim_cur_img_list = []
                    for lr_img, lr_support_patchs, hr_img, name in test_loader:
                        if len(name) == 1:
                            name = name[0]
                        else:
                            raise IOError('testing batch is not 1, please reset!!!')
                        # [1, 3, h, w]
                        hr_img = hr_img.to(device)
                        lr_img = lr_img.to(device)
                        # [1, support_size, 3, h, w]
                        lr_support_patchs = lr_support_patchs.reshape((1,-1,lr_support_patchs.shape[3],lr_support_patchs.shape[4])).to(device)
                        metaNet.eval()
                        metaNet.reset_task_size(task_size = 1)
                        with torch.no_grad():
                            sr_img = metaNet(x=lr_img, support_x=lr_support_patchs)
                            sr_img = sr_img.detach().float().cpu().squeeze(0).numpy().transpose((1, 2, 0))
                            sr_img = sr_img[:,:,(2,1,0)]
                            cv2.imwrite(args.result_checkpoint_dir+args.net_name + '_' + name, sr_img)
                            hr_img = hr_img.detach().float().cpu().squeeze(0).numpy().transpose((1, 2, 0))
                            hr_img = hr_img[:,:,(2,1,0)]

                            psnr = cal_PSNR(hr_img, sr_img)
                            ssim = cal_SSIM(hr_img, sr_img)
                            psnr_cur_img_list.append(psnr)
                            ssim_cur_img_list.append(ssim)

                    psnr = np.array(psnr_cur_img_list).mean()
                    ssim = np.array(ssim_cur_img_list).mean()
                    print('Testing Iteration :  {} PSNR : {} SSIM : {}'.format(str(current_step), str(psnr), str(ssim)))

                    if current_step % args.save_test_psnr_ssim_every == 0:
                        np.savetxt(txt_psnr_path, np.array(test_psnr_visual_list))
                        np.savetxt(txt_ssim_path, np.array(test_ssim_visual_list))
                        plt.xlabel('Iteration')
                        plt.ylabel('PSNR')
                        plt.plot(np.array(test_step_visual_list), np.array(test_psnr_visual_list), label='testing psnr')
                        plt.legend()
                        plt.title('testing psnr')
                        plt.savefig(os.path.join(args.log_checkpoint_dir, 'testing_psnr.png'))
                        plt.close('all')

                        plt.xlabel('Iteration')
                        plt.ylabel('SSIM')
                        plt.plot(np.array(test_step_visual_list), np.array(test_ssim_visual_list), label='testing ssim')
                        plt.legend()
                        plt.title('testing ssim')
                        plt.savefig(os.path.join(args.log_checkpoint_dir, 'testing_ssim.png'))
                        plt.close('all')
                    torch.save(metaNet.state_dict(), os.path.join(args.model_checkpoint_dir, 'CMDSR_'+str(current_step)+'_saved_model.pth'))

                metaNet.train()
                metaNet.reset_task_size(task_size = args.task_size)
                current_step += 1
                if current_step > args.niter:
                    break
if __name__ == '__main__':
    main()