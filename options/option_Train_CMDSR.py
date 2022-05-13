# import template
import argparse
import os
"""
Configuration file
"""
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--gpu_ids', type=str, default='0')
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_condition', type=float, default=1e-4)
    parser.add_argument('--niter', type=int, default=1000000)
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--task_size', type=int, default=8)
    parser.add_argument('--support_size', type=int, default=20)
    parser.add_argument('--milestones', nargs='+', type=int, default=[500000, 900000]) 
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_gamma_condition', type=float, default=0.5)
    parser.add_argument('--coefficient_contrastive_l1', type=float, default=0.1)

    #define for data preprocessing
    parser.add_argument('--is_Ycbcr', type=str2bool, default= False)
    parser.add_argument('--random_crop', type=str2bool, default=True)
    parser.add_argument('--lq_hq_same_size', type=str2bool, default=False)
    parser.add_argument('--hr_crop_size', type=int, default=192)

    #define for degradation preprocessing
    #Gaussian blur
    parser.add_argument('--add_gaussian_blur', type=str2bool, default=True, help='if adding gaussian blur')
    parser.add_argument('--blur_kernel_size', type=int, default=15, help='the kernel size of gaussian blur. same as SRMD.')
    parser.add_argument('--low_blur_sigma', type=float, default=0.2, help='the minimization sigma of gaussian blur')
    parser.add_argument('--range_blur_sigma', type=float, default=3.8, help='the sigma range of gaussian blur')
    parser.add_argument('--add_test_gaussian_blur', type=str2bool, default=True, help='if adding gaussian blur for testing img')
    parser.add_argument('--test_blur_kernel_size', type=int, default=7, help='the kernel size of gaussian blur. same as SRMD.')
    parser.add_argument('--test_blur_sigma', type=float, default=2.6, help='the sigma of gaussian blur for testing img')
    #Noise
    parser.add_argument('--add_noise', type=str2bool, default=True, help='if adding noise')
    parser.add_argument('--noise_level', type=float, default=75, help='noise level to lr_patchs. small for real images, bigger for noisy images and zero for ideal case')
    parser.add_argument('--add_test_noise', type=str2bool, default=True, help='if adding noise for testing img')
    parser.add_argument('--test_noise_level', type=int, default=15, help='noise level to lr_patchs')
    #JPEG
    parser.add_argument('--add_jpeg', type=str2bool, default=False, help='if adding JPEG')
    parser.add_argument('--jpeg_low_quality', type=float, default=30, help='JPEG quality of LR images')
    parser.add_argument('--jpeg_quality_range', type=float, default=50, help='JPEG quality range of LR images')
    parser.add_argument('--add_test_jpeg', type=str2bool, default=False, help='if adding JPEG')
    parser.add_argument('--test_jpeg_quality', type=float, default=60, help='JPEG quality of LR images')


    #define for model
    parser.add_argument('--optmize_sr_condition_net', type=str2bool, default=False)
    parser.add_argument('--opt_condition_net_sep', type=str2bool, default=True)
    parser.add_argument('--opt_condition_net_step', type=int, default=10)
    #Mod
    parser.add_argument('--use_support_Mod', type=str2bool, default=True)
    #sr_net
    parser.add_argument('--input_channels', type=int, default=3, help='the number of input channels for sr net')
    parser.add_argument('--channels', type=int, default=64, help='the number of hidden channels for sr net')
    parser.add_argument('--residual_lr', type=float, default=1.0, help='the lr coefficient of residual connection')
    parser.add_argument('--kernel_size', type=int, default=3, help='the kernel_size of conv')
    parser.add_argument('--n_block', type=int, default=10, help='the number of res-block')
    parser.add_argument('--n_conv_each_block', type=int, default=2, help='the number of conv for each res-block')
    #condition_net
    parser.add_argument('--conv_index', type=str, default='22', help='VGG 22|54')
    parser.add_argument('--group', type=int, default=64, help='the number of group conv')
    
    #define for pre-training
    parser.add_argument('--load_trained_model', type=str2bool, default=False)
    parser.add_argument('--load_trained_model_path', type=str)
    
    parser.add_argument('--use_pretrained_sr_net', type=str2bool, default=False)
    parser.add_argument('--pretrained_sr_net_path', type=str, default='')
    parser.add_argument('--net_name', type=str, default='CMDSR')
    parser.add_argument('--save_metanet_every', type=int, default=5000)

    # define for validation
    parser.add_argument('--print_train_loss_every', type=int, default=10)
    parser.add_argument('--save_train_loss_every', type=int, default=100)
    parser.add_argument('--run_test_every', type=int, default=100)
    parser.add_argument('--start_save', type=int, default=100000)
    parser.add_argument('--save_test_psnr_ssim_every', type=int, default=10)
    parser.add_argument('--self_ensemble_output', type=str2bool, default=False, help='geometric self-ensemble')
    parser.add_argument('--flip_output', type=str2bool, default=False, help='conduct flip for geometric self-ensemble')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--display_test_results', type=str2bool, default=True)
    parser.add_argument('--save_tos_plt', type=str2bool, default=False)
    parser.add_argument('--show_size', type=int, default=512)
    #checkpoint
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./CMDSR/')
    parser.add_argument('--train_hq_file_path', type=str,
                            default='./CMDSR/DIV2K/HR/')
    parser.add_argument('--train_lq_file_path', type=str,
                            default='./CMDSR/DIV2K/LRx4')
    parser.add_argument('--test_hq_file_path', type=str,
                            default='./CMDSR/Set5/X4/HR/')
    parser.add_argument('--test_lq_file_path', type=str,
                            default='./CMDSR/Set5/X4/LRx4_blur_7x7_2.6_gnoise_15/')
    parser.add_argument('--checkpoint_sub_dir', type=str, default='Train_CMDSR')

    args = parser.parse_args()
    #set save losses, psnr, ssim plt
    args.save_train_loss_every = args.save_train_loss_every * args.print_train_loss_every
    args.save_test_psnr_ssim_every = args.save_test_psnr_ssim_every * args.run_test_every
    
    if not os.path.exists(args.train_hq_file_path):
        raise ValueError("Misssing %s"%args.train_hq_file_path)
    else:
        print('%s exists!'%args.train_hq_file_path)
    if not os.path.exists(args.train_lq_file_path):
      raise ValueError("Misssing %s"%args.train_lq_file_path)
    else:
        print('%s exists!'%args.train_lq_file_path)
    if not os.path.exists(args.test_hq_file_path):
        raise ValueError("Misssing %s"%args.test_hq_file_path)
    else:
        print('%s exists!'%args.test_hq_file_path)
    if not os.path.exists(args.test_hq_file_path):
        raise ValueError("Misssing %s"%args.test_hq_file_path)
    else:
        print('%s exists!'%args.test_hq_file_path)
    
    args.checkpoint_dir += args.checkpoint_sub_dir
    args.model_checkpoint_dir = args.checkpoint_dir + '/models/'
    args.result_checkpoint_dir = args.checkpoint_dir + '/results/'
    args.pretrained_model_checkpoint_dir = args.checkpoint_dir + '/pretrained_models/'
    args.log_checkpoint_dir = args.checkpoint_dir + '/log/'

    #create checkpoint dirs
    if os.path.exists(args.checkpoint_dir) and os.path.isfile(args.checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint saving, got a file'.format(
          args.checkpoint_dir))
    elif not os.path.exists(args.checkpoint_dir):
      os.makedirs(args.checkpoint_dir)
      print('%s created successfully!'%args.checkpoint_dir)

    if os.path.exists(args.pretrained_model_checkpoint_dir) and os.path.isfile(args.pretrained_model_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint pretrained model saving, got a file'.format(
          args.pretrained_model_checkpoint_dir))
    elif not os.path.exists(args.pretrained_model_checkpoint_dir):
      os.makedirs(args.pretrained_model_checkpoint_dir)
      print('%s created successfully!'%args.pretrained_model_checkpoint_dir)
    
    if os.path.exists(args.result_checkpoint_dir) and os.path.isfile(args.result_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint results saving, got a file'.format(
          args.result_checkpoint_dir))
    elif not os.path.exists(args.result_checkpoint_dir):
      os.makedirs(args.result_checkpoint_dir)
      print('%s created successfully!'%args.result_checkpoint_dir)
    
    if os.path.exists(args.model_checkpoint_dir) and os.path.isfile(args.model_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint model saving, got a file'.format(
          args.model_checkpoint_dir))
    elif not os.path.exists(args.model_checkpoint_dir):
      os.makedirs(args.model_checkpoint_dir)
      print('%s created successfully!'%args.model_checkpoint_dir)

    if os.path.exists(args.model_checkpoint_dir) and os.path.isfile(args.log_checkpoint_dir):
      raise IOError('Required dst path {} as a directory for checkpoint log saving, got a file'.format(
          args.log_checkpoint_dir))
    elif not os.path.exists(args.log_checkpoint_dir):
      os.makedirs(args.log_checkpoint_dir)
      print('%s created successfully!'%args.log_checkpoint_dir)
    
    return args

if __name__ == "__main__":
    args = set_args()   