from torch.utils.data import Dataset
import os
from copy import deepcopy
import random
import cv2
import numpy as np
import torch
from tools.data_preprocess import add_jpeg, add_noise, Gaussian_blur, fspecial_gaussian

def Get_local_lq_hq_paths(lq_file_path, hq_file_path, shuffle=False):
    img_list = []    
    for img_name in os.listdir(lq_file_path):
        if img_name[-3:] == 'png' or img_name[-3:] == 'jpg' or img_name[-3:] == 'bmp':
            img_list.append(img_name)
    if shuffle:
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        random.shuffle(img_list)
    hq_img_list = deepcopy(img_list) 
    lq_img_list = deepcopy(img_list)
    return lq_img_list, hq_img_list

class SRDataSet(Dataset):
    def __init__(self, args, is_train=True, is_meta=False):
        super(SRDataSet, self).__init__()
        self.args = args
        self.is_train = is_train
        self.is_meta = is_meta
        self.scale_factor = int(args.scale_factor)
        # self.if_Ycbcr = self.args.if_Ycbcr
        self.if_Ycbcr = False
        self.if_test_crop = False
        self.lq_hq_same_size = args.lq_hq_same_size

        if is_train:
            self.hq_file_path = args.train_hq_file_path
            self.lq_file_path = args.train_lq_file_path
        else:
            self.hq_file_path = args.test_hq_file_path
            self.lq_file_path = args.test_lq_file_path
        self.lq_img_list, self.hq_img_list = Get_local_lq_hq_paths(self.lq_file_path, self.hq_file_path)
        if self.args.add_test_gaussian_blur or self.args.add_test_noise or self.args.add_test_jpeg:
            self.lq_file_path = self._create_degradation_file()
        self.dataset_len = self.__len__()


    def _create_degradation_file(self):
        lr_deg_file_name = 'LRx' + str(self.args.scale_factor)
        if self.args.add_test_jpeg:
            jpeg_level = self.args.test_jpeg_quality
            lr_deg_file_name += '_jpeg_{}'.format(str(jpeg_level))
        if self.args.add_test_gaussian_blur:
            blur_kernel_size = self.args.test_blur_kernel_size
            blur_sigma = self.args.test_blur_sigma
            lr_deg_file_name += '_blur_{}x{}_{}'.format(str(self.args.test_blur_kernel_size), str(self.args.test_blur_kernel_size), str(blur_sigma))
            kernel = fspecial_gaussian(blur_kernel_size, blur_sigma)
        if self.args.add_test_noise:
            noise_level = self.args.test_noise_level
            lr_deg_file_name += '_gnoise_{}'.format(str(noise_level))
        
        lq_file_path = os.path.join(self.args.checkpoint_dir,lr_deg_file_name)
        
        if lr_deg_file_name == self.args.test_lq_file_path.split('/')[-2] and os.path.exists(self.args.test_lq_file_path):
            lq_file_path = self.args.test_lq_file_path
            print('{} exists !!!'.format(lq_file_path))
        if os.path.exists(lq_file_path):
            print('{} exists !!!'.format(lq_file_path))
        else:
            os.mkdir(lq_file_path)
            for idx in range(len(self.hq_img_list)):
                hr = cv2.imread(os.path.join(self.hq_file_path, self.hq_img_list[idx]))
                lr = cv2.imread(os.path.join(self.lq_file_path, self.lq_img_list[idx]))
                if self.args.add_test_gaussian_blur:
                    lr = Gaussian_blur(hr, kernel, sf=self.scale_factor)
                if self.args.add_test_noise:
                    lr = add_noise(lr, low_noise_level=0, noise_level_range=noise_level,noise_type="gaussian")
                if self.args.add_test_jpeg:
                    lr,_ = add_jpeg(lr, jpeg_level, None)
                lr_path = os.path.join(lq_file_path, self.lq_img_list[idx])
                cv2.imwrite(lr_path, lr)
        return lq_file_path

    def __getitem__(self, idx):
        # onle lr ,hr  
        lr, hr, filename = self._load_file(idx)

        # print(filename)
        if self.is_train or self.if_test_crop:
        #random patch
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            lr_patch, hr_patch = self._get_pair_patch(lr, hr, hr_size=int(self.args.hr_crop_size))
            lr_patch, hr_patch = self._augment([lr_patch, hr_patch])
        else:
            lr_patch, hr_patch = lr, hr

        #lq hq same size
        if self.args.lq_hq_same_size:
            lr_patch = cv2.resize(lr_patch,(hr_patch.shape[1], hr_patch.shape[0]), interpolation=cv2.INTER_CUBIC)
            # lr_patch = imresize_np(lr_patch/255, self.scale_factor, out_H=hr_patch.shape[0], out_W=hr_patch.shape[1])*255
            if lr_patch.shape[1] != hr_patch.shape[1] or lr_patch.shape[0] != hr_patch.shape[0]:
                print("size error {}".format(filename))
        
        #change cv2 mode from BGR to RGB/YCbCR
        lr_patch, hr_patch = self._set_img_channel([lr_patch, hr_patch], img_mode="RGB")

        #ToTensor
        lr_patch, hr_patch = self._np2Tensor([lr_patch, hr_patch], rgb_range=255)
        if self.is_meta:
            lr_support_patchs = []
            for i in range(self.args.support_size):
                lr_support_patch = self._get_single_patch(lr, int(self.args.hr_crop_size // self.args.scale_factor))
                if self.args.lq_hq_same_size:
                    lr_support_patch = cv2.resize(lr_support_patch, (0,0), fx=self.scale_factor, fy=self.scale_factor)
                lr_support_patch = self._set_img_channel([lr_support_patch], img_mode="RGB")
                lr_support_patch = lr_support_patch[0]
                lr_support_patch = self._np2Tensor([lr_support_patch], rgb_range=255)
                lr_support_patch = lr_support_patch[0]
                lr_support_patchs.append(lr_support_patch.unsqueeze(0))
            lr_support_patchs = torch.cat(lr_support_patchs, 0)
            return lr_patch, lr_support_patchs, hr_patch, filename
        else:
            return lr_patch, hr_patch, filename

    def __len__(self):
        if self.is_train:
            return len(self.hq_img_list)  # * self.repeat
        else:
            return len(self.lq_img_list)

    def _get_index(self, idx):
        if self.is_train:
            return idx % len(self.hq_img_list)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)

        hr = cv2.imread(os.path.join(self.hq_file_path, self.hq_img_list[idx]))
        filename = self.hq_img_list[idx]
        high, width, _ = hr.shape

        lr = cv2.imread(os.path.join(self.lq_file_path, self.lq_img_list[idx]))
        high_lr, width_lr, _ = lr.shape

        if width_lr != width // self.scale_factor or high_lr != high // self.scale_factor:
            lr = cv2.resize(lr, (int(width // self.scale_factor), int(high // self.scale_factor)), interpolation=cv2.INTER_CUBIC)
        return lr, hr, filename


    def _get_pair_patch(self, lr, hr, hr_size=512):
        ih, iw = lr.shape[:2]
        # print(ih,iw)

        tp = int(hr_size)
        ip = int (tp // self.scale_factor)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        tx, ty = self.scale_factor * ix, self.scale_factor * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        hr_patch = hr[ty:ty + tp, tx:tx + tp, :]

        return lr_patch, hr_patch

    def _get_pair_same_patch(self, lr, hr, hr_size=512):
        if lr.shape != hr.shape:
            raise IOError('the shapes of lr and hr are different!!!')
        sh, sw = lr.shape[:2]
        sp = int(hr_size)

        sx = random.randrange(0, sw - sp + 1)
        sy = random.randrange(0, sh - sp + 1)

        lr_patch = lr[sy:sy + sp, sx:sx + sp, :]
        hr_patch = hr[sy:sy + sp, sx:sx + sp, :]

        return lr_patch, hr_patch
    
    def _get_single_patch(self, lr, lr_size=128):
        ih, iw = lr.shape[:2]

        if self.args.is_train:
            ix = random.randrange(0, iw - lr_size + 1)
            iy = random.randrange(0, ih - lr_size + 1)
        else:ix,iy=(iw - lr_size + 1)//2,(ih - lr_size + 1)//2

        lr_patch = lr[iy:iy + lr_size, ix:ix + lr_size, :]

        return lr_patch
    
    def _augment(self, l, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _single_augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            
            return img

        return [_single_augment(_l) for _l in l]

    def _set_img_channel(self, l, img_mode="RGB"):
        def _set_single_img_channel(img, img_mode):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            c = img.shape[2]
            if img_mode == "YCbCr" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif img_mode == "YCbCr" and c == 1:
                img = np.concatenate([img] * 3, 2)
            if img_mode == "RGB" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        return [_set_single_img_channel(_l, img_mode) for _l in l]

    def _np2Tensor(self, l, rgb_range):
        def _single_np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)

            return tensor

        return [_single_np2Tensor(_l) for _l in l]

class SRMultiGroupRandomTaskDataSet(Dataset):
    def __init__(self, args, is_train=True):
        super(SRMultiGroupRandomTaskDataSet, self).__init__()
        self.args = args
        self.is_train = is_train
        # self.if_Ycbcr = self.args.if_Ycbcr
        self.if_Ycbcr = False
        self.lq_hq_same_size = args.lq_hq_same_size
        self.scale_factor = int(args.scale_factor)
        self.support_size = args.support_size

        self.degradation_coefficient = None
        if is_train:
            self.hq_file_path = args.train_hq_file_path
            self.lq_file_path = args.train_lq_file_path
        else:
            self.hq_file_path = args.test_hq_file_path
            self.lq_file_path = args.test_lq_file_path
        self.lq_img_list, self.hq_img_list = Get_local_lq_hq_paths(self.lq_file_path, self.hq_file_path)
        
        self.dataset_len = self.__len__()

    def random_degradation_param(self):
        blur_kernel_size, blur_sigma, noise_level, jpeg_quality = None, None, None, None
        deg_cof = "deg"
        #generate degradation cofficients
        if self.args.add_gaussian_blur:
            blur_kernel_size = (self.args.blur_kernel_size, self.args.blur_kernel_size)
            rand_range = int(self.args.range_blur_sigma*10)
            blur_sigma = round(self.args.low_blur_sigma + random.randint(0, rand_range)/10, 2)
            deg_cof += "_blur_{}".format(str(blur_sigma))
        if self.args.add_noise:
            noise_level = int(random.randint(0,self.args.noise_level))
            deg_cof += "_noise_{}".format(str(noise_level))
        if self.args.add_jpeg:
            jpeg_quality = int(random.randint(self.args.jpeg_low_quality,self.args.jpeg_low_quality+self.args.jpeg_quality_range))
            deg_cof += "_jpeg_{}".format(str(jpeg_quality))
        return deg_cof, blur_kernel_size, blur_sigma, noise_level, jpeg_quality

    def random_degradation_transfer(self, deg_cof, blur_kernel_size, blur_sigma, noise_level, jpeg_quality, bi_lr_patch, hr_patch):
        # [h, w, c]
        lr_deg_patch = deepcopy(bi_lr_patch)
        if self.args.add_gaussian_blur:
            lr_patch_blur = cv2.GaussianBlur(hr_patch, blur_kernel_size, blur_sigma)
            lr_deg_patch = cv2.resize(lr_patch_blur,(lr_deg_patch.shape[1], lr_deg_patch.shape[0]), interpolation=cv2.INTER_CUBIC) 
        if self.args.add_noise:
            lr_deg_patch = add_noise(lr_deg_patch, low_noise_level=0, noise_level_range=noise_level,noise_type="gaussian")
        if self.args.add_jpeg:
            lr_deg_patch, jpeg_quality = add_jpeg(lr_deg_patch, jpeg_quality, None)
        
        #lq hq same size
        if self.args.lq_hq_same_size:
            lr_deg_patch = cv2.resize(lr_deg_patch,(hr_patch.shape[1], hr_patch.shape[0]), interpolation=cv2.INTER_CUBIC)
            if lr_deg_patch.shape[1] != hr_patch.shape[1] or lr_deg_patch.shape[0] != hr_patch.shape[0]:
                print("size error")
        
        return bi_lr_patch, lr_deg_patch, hr_patch, deg_cof

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)

        bi_lr_patch_tensors, lr_patch_tensors, hr_patch_tesnors, filenames = [], [], [], []
        
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        
        deg_cof, blur_kernel_size, blur_sigma, noise_level, jpeg_quality = self.random_degradation_param()

        for i in range(self.support_size):
            bi_lr_patch, hr_patch = self._get_pair_patch(lr, hr, hr_size=int(self.args.hr_crop_size))
            bi_lr_patch, lr_patch, hr_patch, deg_cof = self.random_degradation_transfer(deg_cof, blur_kernel_size, blur_sigma, noise_level, jpeg_quality,bi_lr_patch, hr_patch)
            bi_lr_patch, lr_patch, hr_patch = self._augment([bi_lr_patch, lr_patch, hr_patch])
            #change cv2 mode from BGR to RGB
            bi_lr_patch, lr_patch, hr_patch = self._set_img_channel([bi_lr_patch, lr_patch, hr_patch], img_mode="RGB")
            #ToTensor
            bi_lr_patch, lr_patch, hr_patch = self._np2Tensor([bi_lr_patch, lr_patch, hr_patch], rgb_range=255)
            
            bi_lr_patch_tensors.append(bi_lr_patch.unsqueeze(0))
            lr_patch_tensors.append(lr_patch.unsqueeze(0))
            hr_patch_tesnors.append(hr_patch.unsqueeze(0))
            filenames.append(filename)

        bi_lr_patch_tensors = torch.cat(bi_lr_patch_tensors, 0)
        lr_patch_tensors = torch.cat(lr_patch_tensors,0)
        hr_patch_tesnors = torch.cat(hr_patch_tesnors,0)

        return lr_patch_tensors, hr_patch_tesnors, filenames, deg_cof
        
    def __len__(self):
        if self.is_train:
            return len(self.hq_img_list)  # * self.repeat
        else:
            return len(self.lq_img_list)

    def _get_index(self, idx):
        if self.is_train:
            return idx % len(self.hq_img_list)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)

        hr = cv2.imread(os.path.join(self.hq_file_path, self.hq_img_list[idx]))
        filename = self.hq_img_list[idx]
        high, width, _ = hr.shape

        # print('f_hr:', f_hr)
        # print('f_lrin:', f_lrin)

        lr = cv2.imread(os.path.join(self.lq_file_path, self.lq_img_list[idx]))
        high_lr, width_lr, _ = lr.shape

        if width_lr != width // self.scale_factor or high_lr != high // self.scale_factor:
            lr = cv2.resize(lr, (int(width // self.scale_factor), int(high // self.scale_factor)), interpolation=cv2.INTER_CUBIC)
        return lr, hr, filename


    def _get_pair_patch(self, lr, hr, hr_size=512):
        ih, iw = lr.shape[:2]
        # print(ih,iw)

        tp = int(hr_size)
        ip = int (tp // self.scale_factor)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        tx, ty = self.scale_factor * ix, self.scale_factor * iy

        lr_patch = lr[iy:iy + ip, ix:ix + ip, :]
        hr_patch = hr[ty:ty + tp, tx:tx + tp, :]

        return lr_patch, hr_patch

    def _get_pair_same_patch(self, lr, hr, hr_size=512):
        if lr.shape != hr.shape:
            raise IOError('the shapes of lr and hr are different!!!')
        sh, sw = lr.shape[:2]
        sp = int(hr_size)

        sx = random.randrange(0, sw - sp + 1)
        sy = random.randrange(0, sh - sp + 1)

        lr_patch = lr[sy:sy + sp, sx:sx + sp, :]
        hr_patch = hr[sy:sy + sp, sx:sx + sp, :]

        return lr_patch, hr_patch
    
    def _get_single_patch(self, lr, lr_size=128):
        ih, iw = lr.shape[:2]

        if self.args.is_train:
            ix = random.randrange(0, iw - lr_size + 1)
            iy = random.randrange(0, ih - lr_size + 1)
        else:ix,iy=(iw - lr_size + 1)//2,(ih - lr_size + 1)//2

        lr_patch = lr[iy:iy + lr_size, ix:ix + lr_size, :]

        return lr_patch
    
    def _augment(self, l, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _single_augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            
            return img

        return [_single_augment(_l) for _l in l]

    def _set_img_channel(self, l, img_mode="RGB"):
        def _set_single_img_channel(img, img_mode):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            c = img.shape[2]
            if img_mode == "YCbCr" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif img_mode == "YCbCr" and c == 1:
                img = np.concatenate([img] * 3, 2)
            if img_mode == "RGB" and c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        return [_set_single_img_channel(_l, img_mode) for _l in l]

    def _np2Tensor(self, l, rgb_range):
        def _single_np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)

            return tensor

        return [_single_np2Tensor(_l) for _l in l]
    


 
   
