from options.option_Test_CMDSR_single_img import set_args
from tools.utils import check_args
from models.CMDSR import CMDSR
import torch
import numpy as np
import cv2
import random

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

    metaNet.eval()
    metaNet.reset_task_size(task_size=1)

    lr = cv2.imread(args.lr_path)
    lr_img_RGB = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    # _np2Tensor
    lr_img = np.ascontiguousarray(lr_img_RGB.transpose((2, 0, 1)))
    lr_img = torch.from_numpy(lr_img).float()
    lr_support_patchs = []
    for i in range(args.support_size):
        # _get_single_patch
        lr_size = int(args.hr_crop_size // args.scale_factor)
        ih, iw = lr_img_RGB.shape[:2]
        ix = random.randrange(0, iw - lr_size + 1)
        iy = random.randrange(0, ih - lr_size + 1)
        lr_support_patch = lr[iy:iy + lr_size, ix:ix + lr_size, :]
        #_np2Tensor
        lr_support_patch = np.ascontiguousarray(lr_support_patch.transpose((2, 0, 1)))
        lr_support_patch = torch.from_numpy(lr_support_patch).float()
        lr_support_patchs.append(lr_support_patch.unsqueeze(0))
    lr_support_patchs = torch.cat(lr_support_patchs, 0)
    #to device
    lr_img = lr_img.unsqueeze(0).to(device)
    lr_support_patchs = lr_support_patchs.reshape((1,-1,lr_support_patchs.shape[2],lr_support_patchs.shape[3])).to(device)
    with torch.no_grad():
        sr_img = metaNet(x=lr_img, support_x=lr_support_patchs)
        sr_img = sr_img.detach().float().cpu().squeeze(0).numpy().transpose((1, 2, 0))
        sr_img = sr_img[:,:,(2,1,0)]
        print('Saving SR image to {}'.format(args.result_checkpoint_dir))
        cv2.imwrite(args.result_checkpoint_dir+args.lr_path.split('/')[-1], sr_img)
        
                    
if __name__ == '__main__':
    main()