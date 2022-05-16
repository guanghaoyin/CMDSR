from options.option_Test_CMDSR import set_args
from tools.utils import check_args
from data_loader import SRDataSet
from models.CMDSR import CMDSR
import torch
from torch.utils.data import DataLoader
from PSNR_SSIM_cal import cal_PSNR, cal_SSIM
import numpy as np
import cv2


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
    #dataset
    test_dataset = SRDataSet(args=args, is_train=False, is_meta=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    
    print('The nums of testing images {:,d}'.format(len(test_loader)))

    print('Start testing!')
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
        with torch.no_grad():
            sr_img = metaNet(x=lr_img, support_x=lr_support_patchs)
            sr_img = sr_img.detach().float().cpu().squeeze(0).numpy().transpose((1, 2, 0))
            sr_img = sr_img[:,:,(2,1,0)]
            cv2.imwrite(args.result_checkpoint_dir+args.net_name+ '_' + name, sr_img)
            hr_img = hr_img.detach().float().cpu().squeeze(0).numpy().transpose((1, 2, 0))
            hr_img = hr_img[:,:,(2,1,0)]

            psnr = cal_PSNR(hr_img, sr_img)
            ssim = cal_SSIM(hr_img, sr_img)
            psnr_cur_img_list.append(psnr)
            ssim_cur_img_list.append(ssim)
            print('Testing Img : {} PSNR : {} SSIM : {}'.format(name, str(psnr), str(ssim)))

    psnr = np.array(psnr_cur_img_list).mean()
    ssim = np.array(ssim_cur_img_list).mean()
    print('Testing Ave_PSNR : {} Ave_SSIM : {}'.format(str(psnr), str(ssim)))

                    
if __name__ == '__main__':
    main()