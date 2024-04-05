import math
import cv2
import numpy as np
import lpips
import torch

#### PSNR
def img_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

#### SSIM
def img_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


#### LPIPS

# https://github.com/richzhang/PerceptualSimilarity
loss_fn = lpips.LPIPS(net='alex', spatial=False).cuda() # Can also set net = 'squeeze' or 'vgg'

def img_lpips(img1, img2):

    def process(img):
        img = torch.from_numpy(img)[:,:,[2,1,0]].float()
        return img.permute(2,0,1).unsqueeze(0).cuda() * 2 - 1

    img1 = process(img1)
    img2 = process(img2)
    return loss_fn.forward(img1, img2).mean().detach().cpu().tolist()

#### LOE
def img_loe(ipic, epic, window_size=7):

    def U_feature(image):
        image = cv2.resize(image, (500,500))
        image = np.max(image, axis=2)
        w_half = window_size // 2
        padded_arr = np.pad(image, ((w_half, w_half), (w_half, w_half)), mode='constant')

        local_windows = np.lib.stride_tricks.sliding_window_view(padded_arr, (window_size, window_size))
        local_windows = local_windows.reshape(-1, window_size * window_size)
        relationship = local_windows[:,:,None] > local_windows[:,None,:]
        return relationship.flatten()

    ipic = U_feature(ipic)
    epic = U_feature(epic)

    return np.mean(ipic!=epic)

def metric(gt_image_path, pred_image_path):
    gt_image = cv2.imread(gt_image_path) / 255.
    pred_image = cv2.imread(pred_image_path) / 255.
    pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

    psnr = img_psnr(gt_image, pred_image)
    ssim = img_ssim(gt_image, pred_image)
    lpips = img_lpips(gt_image, pred_image)
    loe = img_loe(gt_image, pred_image)
