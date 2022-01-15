import os
import numpy as np
from PIL import Image
import cv2
import torch
from datetime import datetime
import sys
import math
from torchvision.utils import make_grid
import torchvision.transforms as transforms

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def img2tensor(image,args,is_Crop=False,crop_size=256):
    # opencv image to PIL image
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform_list = []
    if is_Crop:
        transform_list.append(transforms.RandomCrop(crop_size))
    transform_list.append(transforms.ToTensor())
    if args.is_normalize_datas:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # to [-1,1]
    img2tensor_t = transforms.Compose(transform_list)
    out_tensor = img2tensor_t(img)
    return out_tensor
# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self, args):
        self.terminal = sys.stdout
        args.path_log = args.checkpoint_dir + 'print_log.txt' if args.use_docker else 'print_log.txt'
        self.log = open(args.path_log, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def check_args(args, rank=0):
    if rank == 0:
        # if args.use_docker:
        #     args.setting_file = args.checkpoint_dir + args.setting_file
        #     args.log_file = args.checkpoint_dir + args.log_file
        #     # os.makedirs(args.training_state, exist_ok=True)
        #     os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(args.setting_file, 'w') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')

    return args


def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)
    h,w,_=img.shape
    #img = cv2.resize(img,(int(w/2),int(h/2)))

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def tensor2Image_test(input_image,args,imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.squeeze().cpu().float().numpy()
    return numpy2im(image_numpy, args,imtype,outtype='numpy')

# utils
def tensor2im(input_image, args, imtype=np.uint8, show_size=None):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    im = []
    for i in range(image_numpy.shape[0]):
        if show_size:
            im.append(
                np.array(numpy2im(image_numpy[i], args,imtype).resize((show_size, show_size), Image.ANTIALIAS)))
        else:
            im.append(np.array(numpy2im(image_numpy[i], args,imtype)))
    return np.array(im)


def numpy2im(image_numpy, args,imtype=np.uint8,outtype='PIL'):
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    if args.is_normalize_datas:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) / 2. + 0.5) * 255.0
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = image_numpy.astype(imtype)
    if outtype=='PIL':
        return Image.fromarray(image_numpy)
    else:
        return image_numpy

def display_online_cv2img_results(visuals, args, img_name, vis_saved_dir, show_size=128, in_channel=3):
    pass


def display_online_results(visuals, args,steps, vis_saved_dir, show_size=128, in_channel=3):
    images = []
    labels = []
    for label, image in visuals.items():
        image_numpy = tensor2im(image, args,show_size=show_size)  # [10, 128, 128, 3]
        image_numpy = np.reshape(image_numpy, (-1, show_size, in_channel))
        images.append(image_numpy)
        labels.append(label)
    save_images = np.array(images)  # [8, 128*10, 128, 3]
    save_images = np.transpose(save_images, [1, 0, 2, 3])
    save_images = np.reshape(save_images, (save_images.shape[0], -1, in_channel))
    title_img = get_title(labels, show_size,in_channel=in_channel)
    save_images = np.concatenate([title_img, save_images], axis=0)
    img_save_path = os.path.join(vis_saved_dir, 'display_' + str(steps) + '.png')
    save_image(save_images, img_save_path)
    save_images = save_images[:, :, (2, 1, 0)]
    data = np.array(cv2.imencode('.png', save_images)[1]).tostring()
    return img_save_path,data


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(np.squeeze(image_numpy))
    image_pil.save(image_path,quality=100)


def get_title(labels, show_size=128, in_channel=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_img = []
    for label in labels:
        x = np.ones((40, show_size, in_channel)) * 255.0
        textsize = cv2.getTextSize(label, font, 0.5, 2)[0]
        x = cv2.putText(x, label, ((x.shape[1] - textsize[0]) // 2, x.shape[0] // 2), font, 0.5, (0, 0, 0), 1)
        title_img.append(x)

    title_img = np.array(title_img)
    title_img = np.transpose(title_img, [1, 0, 2, 3])
    title_img = np.reshape(title_img, [title_img.shape[0], -1, in_channel])
    title_img = title_img.astype(np.uint8)

    return title_img

def read_img(path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2rgb(img):
    '''bgr2rgb
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    if in_img_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

import random

def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if ss == target_width:
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == "scale_shortside_and_crop":
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {"crop_pos": (x, y), "flip": flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if "resize" in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif "scale_width" in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif "scale_shortside" in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if "crop" in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params["crop_pos"], opt.crop_size)))

    if opt.preprocess_mode == "none":
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == "fixed":
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.is_train:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params["flip"])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
