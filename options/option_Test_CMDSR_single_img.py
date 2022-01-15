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
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--task_size', type=int, default=1)
    parser.add_argument('--support_size', type=int, default=20)

    #define for data preprocessing
    parser.add_argument('--is_Ycbcr', type=str2bool, default= False)
    parser.add_argument('--random_crop', type=str2bool, default=True)
    parser.add_argument('--lq_hq_same_size', type=str2bool, default=False)
    parser.add_argument('--hr_crop_size', type=int, default=128)

    #define for degradation preprocessing
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
    
    #define for  pre-training and loading trained model
    parser.add_argument('--load_trained_model', type=str2bool, default=True)
    parser.add_argument('--load_trained_model_path', type=str, default='./model_zoo/CMDSR_model.pth')
    
    parser.add_argument('--use_pretrained_sr_net', type=str2bool, default=False)
    parser.add_argument('--pretrained_sr_net_path', type=str, default='None')
    parser.add_argument('--net_name', type=str, default='CMDSR')

    # define for validation
    parser.add_argument('--self_ensemble_output', type=str2bool, default=False, help='geometric self-ensemble')
    parser.add_argument('--flip_output', type=str2bool, default=False, help='conduct flip for geometric self-ensemble')
    parser.add_argument('--use_hdfs_movie_eval', type=str2bool, default=False)
    parser.add_argument('--log_file', type=str, default='test_log.txt')
    parser.add_argument('--setting_file', type=str, default='test_setting.txt')
    parser.add_argument('--display_test_results', type=str2bool, default=True)
    parser.add_argument('--save_tos_plt', type=str2bool, default=False)
    parser.add_argument('--show_size', type=int, default=512)
    
    #checkpoint
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./CMDSR/',
                        help='first save in the docker, then upload to hdfs')
    parser.add_argument('--checkpoint_sub_dir', type=str, default='Test_CMDSR_single_img')
    
    #LR path
    parser.add_argument('--lr_path', type=str,
                            default='./Real_img/Flowers.png')

    args = parser.parse_args()

    if not os.path.exists(args.lr_path):
      raise ValueError("Misssing %s"%args.lr_path)
    #set save losses, psnr, ssim plt
    args.checkpoint_dir += args.checkpoint_sub_dir
    # args.setting_file = args.checkpoint_dir + args.setting_file
    args.model_checkpoint_dir = args.checkpoint_dir + '/models/'
    args.result_checkpoint_dir = args.checkpoint_dir + '/results/'
    args.pretrained_model_checkpoint_dir = args.checkpoint_dir + '/pretrained_models/'
    args.log_checkpoint_dir = args.checkpoint_dir + '/log/'

    #create checkpoint_dir
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
