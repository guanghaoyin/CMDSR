import cv2
import numpy as np
import math
from tools.utils import bgr2ycbcr

def cal_PSNR(img_HR, img_SR, use_YCbCr = True):
    #x3
    # if img_SR.shape[1] != img_HR.shape[1] or img_SR.shape[0] != img_HR.shape[0]:
        # img_HR = cv2.resize(img_HR,(img_SR.shape[1], img_SR.shape[0]), interpolation=cv2.INTER_CUBIC)
    # crop borders
    crop_border = 4
    if img_HR.ndim == 3:
        img_HR = img_HR[crop_border:-crop_border, crop_border:-crop_border, :]
        img_SR = img_SR[crop_border:-crop_border, crop_border:-crop_border, :]
    elif img_HR.ndim == 2:
        img_HR = img_HR[crop_border:-crop_border, crop_border:-crop_border]
        img_SR = img_SR[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(img_HR.ndim))
    if use_YCbCr:
        img_HR = bgr2ycbcr(img_HR/255,only_y=True)
        img_HR = img_HR * 255
        img_SR = bgr2ycbcr(img_SR/255,only_y=True)
        img_SR = img_SR * 255
    
    # img_SR and img_HR have range [0, 255]
    img_SR = img_SR.astype(np.float64)
    img_HR = img_HR.astype(np.float64)
    mse = np.mean((img_SR - img_HR)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img_SR, img_HR):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img_SR = img_SR.astype(np.float64)
    img_HR = img_HR.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img_SR, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img_HR, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img_SR**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img_HR**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img_SR * img_HR, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def cal_SSIM(img_HR, img_SR, use_YCbCr=True):
    '''calculate SSIM
    the same outputs as MATLAB's
    img_SR, img_HR: [0, 255]
    '''
    #x3
    # if img_SR.shape[1] != img_HR.shape[1] or img_SR.shape[0] != img_HR.shape[0]:
        # img_HR = cv2.resize(img_HR,(img_SR.shape[1], img_SR.shape[0]), interpolation=cv2.INTER_CUBIC)
    crop_border = 4
    if img_HR.ndim == 3:
        img_HR = img_HR[crop_border:-crop_border, crop_border:-crop_border, :]
        img_SR = img_SR[crop_border:-crop_border, crop_border:-crop_border, :]
    elif img_HR.ndim == 2:
        img_HR = img_HR[crop_border:-crop_border, crop_border:-crop_border]
        img_SR = img_SR[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(img_HR.ndim))
    if use_YCbCr:
        img_HR = bgr2ycbcr(img_HR/255,only_y=True)
        img_HR = img_HR * 255
        img_SR = bgr2ycbcr(img_SR/255,only_y=True)
        img_SR = img_SR * 255
    if not img_SR.shape == img_HR.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img_SR.ndim == 2:
        return ssim(img_SR, img_HR)
    elif img_SR.ndim == 3:
        if img_SR.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img_SR, img_HR))
            return np.array(ssims).mean()
        elif img_SR.shape[2] == 1:
            return ssim(np.squeeze(img_SR), np.squeeze(img_HR))
    else:
        raise ValueError('Wrong input image dimensions.')
if __name__ == '__main__':
    img_HR = cv2.imread('/opt/tiger/lab/yinguanghao/MetaSR/Set5/HR/img_001_SRF_4_HR.png')
    img_LR = cv2.imread('/opt/tiger/lab/yinguanghao/MetaSR/Set5/LRx4/img_001_SRF_4_LR.png')
    img_SR = cv2.resize(img_LR,(0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    print('PSNR:{}'.format(cal_PSNR(img_HR, img_SR, True)))
    print('SSIM:{}'.format(cal_SSIM(img_HR, img_SR, True)))