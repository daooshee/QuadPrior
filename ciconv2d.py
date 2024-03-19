# Import general dependencies
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cldm.model import create_model, load_state_dict

eps = 1e-4

# ==================================
# ======== Gaussian filter =========
# ==================================

# https://github.com/Attila94/CIConv/blob/main/method/ciconv2d.py

def gaussian_basis_filters(scale, gpu, k=3):
    std = torch.pow(2,scale)

    # Define the basis vector for the current scale
    filtersize = torch.ceil(k*std+0.5)
    x = torch.arange(start=-filtersize.item(), end=filtersize.item()+1)
    if gpu is not None: x = x.to(gpu); std = std.to(gpu)
    x = torch.meshgrid([x,x])

    # Calculate Gaussian filter base
    # Only exponent part of Gaussian function since it is normalized anyway
    g = torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    g = g / torch.sum(g)  # Normalize

    # Gaussian derivative dg/dx filter base
    dgdx = -x[0]/(std**3*2*math.pi)*torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    dgdx = dgdx / torch.sum(torch.abs(dgdx))  # Normalize

    # Gaussian derivative dg/dy filter base
    dgdy = -x[1]/(std**3*2*math.pi)*torch.exp(-(x[1]/std)**2/2)*torch.exp(-(x[0]/std)**2/2)
    dgdy = dgdy / torch.sum(torch.abs(dgdy))  # Normalize

    # Stack and expand dim
    basis_filter = torch.stack([g,dgdx,dgdy], dim=0)[:,None,:,:]

    return basis_filter

def convolve_gaussian_filters(batch, scale):
    E, El, Ell = torch.split(batch, 1, dim=1)
    E_out, El_out, Ell_out = [], [], []

    for s in range(len(scale)):
        # Convolve with Gaussian filters
        w = gaussian_basis_filters(scale=scale[s:s+1], gpu=batch.device).to(dtype=batch.dtype)  # KCHW

        # the padding here works as "same" for odd kernel sizes
        E_out.append(F.conv2d(input=E[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        El_out.append(F.conv2d(input=El[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        Ell_out.append(F.conv2d(input=Ell[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))

    return torch.cat(E_out), torch.cat(El_out), torch.cat(Ell_out)

# =================================
# == Color invariant definitions ==
# =================================

# def E_inv(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
#     E = Ex**2+Ey**2+Elx**2+Ely**2+Ellx**2+Elly**2
#     return E

# def W_inv(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
#     Wx = Ex/(E+eps)
#     Wlx = Elx/(E+eps)
#     Wllx = Ellx/(E+eps)
#     Wy = Ey/(E+eps)
#     Wly = Ely/(E+eps)
#     Wlly = Elly/(E+eps)

#     W = Wx**2+Wy**2+Wlx**2+Wly**2+Wllx**2+Wlly**2
#     return W

# def C_inv(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
#     Clx = (Elx*E-El*Ex)/(E**2+1e-5)
#     Cly = (Ely*E-El*Ey)/(E**2+1e-5)
#     Cllx = (Ellx*E-Ell*Ex)/(E**2+1e-5)
#     Clly = (Elly*E-Ell*Ey)/(E**2+1e-5)

#     C = Clx**2+Cly**2+Cllx**2+Clly**2
#     return C

# def N_inv(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
#     Nlx = (Elx*E-El*Ex)/(E**2+1e-5)
#     Nly = (Ely*E-El*Ey)/(E**2+1e-5)
#     Nllx = (Ellx*E**2-Ell*Ex*E-2*Elx*El*E+2*El**2*Ex)/(E**3+1e-5)
#     Nlly = (Elly*E**2-Ell*Ey*E-2*Ely*El*E+2*El**2*Ey)/(E**3+1e-5)

#     N = Nlx**2+Nly**2+Nllx**2+Nlly**2
#     return N

def H_inv(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    Hx = (Ell*Elx-El*Ellx)/(El**2+Ell**2+1e-5)
    Hy = (Ell*Ely-El*Elly)/(El**2+Ell**2+1e-5)
    H = Hx**2+Hy**2
    return H

# http://103.133.35.64:8080/jspui/bitstream/123456789/642/1/Color%20in%20Computer%20Vision%20-%202012%20-%20Gevers.pdf

def hat_H(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    return torch.atan(El/(Ell+eps))

def hat_S(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    return (El**2 + Ell**2) / (E**2 +eps) 

# def hat_C(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
#     return El/(E+eps)

# def hat_Hx(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
#     return (Ell*Elx-El*Ellx)/(El**2+Ell**2+eps)

# def hat_Hy(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
#     return (Ell*Ely-El*Elly)/(El**2+Ell**2+eps)

def hat_Ww(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    Wx = Ex / (E+eps)
    Wy = Ey / (E+eps)
    return Wx**2 + Wy**2

def hat_Wlw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    Wlx = Elx / (E+eps)
    Wly = Ely / (E+eps)
    return Wlx**2 + Wly**2

def hat_Wllw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    Wllx = Ellx / (E+eps)
    Wlly = Elly / (E+eps)
    return Wllx**2 + Wlly**2

# =================================
# == Color invariant convolution ==
# =================================

class PriorConv2d(nn.Module):
    def __init__(self, invariant, k=3, scale=0.0):

        super(PriorConv2d, self).__init__()
        self.use_cuda = torch.cuda.is_available()

        # Constants
        # self.gcm = torch.tensor([[0.06,0.63,0.27],[0.3,0.04,-0.35],[0.34,-0.6,0.17]])
        self.gcm = torch.nn.Parameter(torch.tensor([[0.06,0.63,0.27],[0.3,0.04,-0.35],[0.34,-0.6,0.17]]))
        self.k = k
            
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1)
        )

        self.mask_Ww_p = 0.5

    def forward(self, batch):
        # Make sure scale does not explode: clamp to max abs value of 2.5
        # self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)

        with torch.no_grad():
            max_RGB = torch.argmax(batch, dim=1)
            min_RGB = torch.argmin(batch, dim=1)

            batch_ = torch.flip(batch, dims=(1,))

            max_RGB_ = 2 - torch.argmax(batch_, dim=1)
            min_RGB_ = 2 - torch.argmin(batch_, dim=1)

            RGB_order = torch.zeros(batch.shape, device=batch.device, dtype=batch.dtype)
            RGB_order = RGB_order.scatter_(1, max_RGB.unsqueeze(1), 0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, max_RGB_.unsqueeze(1), 0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, min_RGB.unsqueeze(1), -0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, min_RGB_.unsqueeze(1), -0.5, reduce='add')

        scale = torch.mean(self.conv(batch), dim=(1,2,3))
        scale = torch.clamp(scale, min=-2.5, max=2.5) 

        # Measure E, El, Ell by Gaussian color model
        in_shape = batch.shape  # bchw
        batch = batch.view((in_shape[:2]+(-1,)))  # flatten image
        batch = torch.matmul(self.gcm.to(batch.device, dtype=batch.dtype), batch)  # estimate E,El,Ell
        batch = batch.view((in_shape[0],)+(3,)+in_shape[2:])  # reshape to original image size

        E_out, El_out, Ell_out = convolve_gaussian_filters(batch.float(), scale.float())

        if False:
            print("Ws")
            E, Ex, Ey = torch.split(E_out,1,dim=1)
            El, Elx, Ely = torch.split(El_out,1,dim=1)
            Ell, Ellx, Elly = torch.split(Ell_out,1,dim=1)

            H = hat_H(E,Ex,Ey,El,None,None,Ell,None,None)
            S = torch.log(hat_S(E,Ex,Ey,El,None,None,Ell,None,None)+eps)
            Ww = torch.atan(hat_Ww(E,Ex,Ey,El,None,None,Ell,None,None) + \
                            hat_Wlw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly) + \
                            hat_Wllw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly))
        else:
            # print("W1")
            E, Ex, Ey = torch.split(E_out,1,dim=1)
            El = torch.split(El_out,1,dim=1)[0]
            Ell = torch.split(Ell_out,1,dim=1)[0]

            H = hat_H(E,Ex,Ey,El,None,None,Ell,None,None)
            S = torch.log(hat_S(E,Ex,Ey,El,None,None,Ell,None,None)+eps)
            Ww = torch.atan(hat_Ww(E,Ex,Ey,El,None,None,Ell,None,None))
            # Wlw2 = hat_Wlw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly)
            # Wllw2 = hat_Wllw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly)

        # if random.random() < self.mask_Ww_p:
        #     Ww = Ww * 0

        features = torch.cat([H, S, RGB_order, Ww], dim=1)
        return features.to(dtype=batch.dtype)


if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    import torch
    from PIL import Image
    import io
    import random
    import os
    import copy
    import glob
    import numpy as np
    import cv2
    
    from matplotlib import pyplot as plt
    
    layer = PriorConv2d("E").cuda().to(dtype=torch.float16)
    image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()]
        )
    
    checkpoint_file = './checkpoints/COCO-learnable_H_S_RGB_Order-Clean-T0Loss-30k.ckpt'
    save_name = checkpoint_file.split("/")[-1].replace(".ckpt", "")
    from cldm.model import create_model, load_state_dict
    state_dict = load_state_dict(checkpoint_file, location='cpu')
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if '_forward_module.control_model.prior_conv' in sd_name:
            new_state_dict[sd_name.replace('_forward_module.control_model.prior_conv.', '')] = sd_param
    layer.load_state_dict(new_state_dict)

    for img_path in glob.glob("/data06/v-wenjwang/LIME/10.bmp"):
        name = img_path.split("/")[-1]

        image = Image.open(img_path).convert('RGB')
        image = image_transform(image).unsqueeze(0).cuda().to(dtype=torch.float16)

        # layer(image)

        RGB_order = layer(image)
        # print(feature.min(), feature.max(), feature.shape)
        RGB_order = torch.transpose(RGB_order, 1, 2)
        RGB_order = torch.transpose(RGB_order, 2, 3)
        RGB_order[RGB_order > 1e-5] = 255
        RGB_order[RGB_order != 255] = 0
        RGB_order = 255-RGB_order

        # # H = (H.cpu().data.numpy()[0,:,:,:]+9.22)/11.*255.
        # RGB_order = ((RGB_order.cpu().data.numpy()[0,:,:,:]+9.3)*255/11.3) # .astype(np.uint8)
        RGB_order = ((RGB_order.cpu().data.numpy()[0,:,:,:])) # .astype(np.uint8)
        # print(RGB_order.min(), RGB_order.max(), RGB_order.shape, type(RGB_order))

        cv2.imwrite("Ww=1-"+name, RGB_order)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(H, cmap=plt.cm.hot_r)
        # plt.savefig("draw-alpha/60000-"+name)
        # plt.close()

    # H, S, Ww, Wlw2, Wllw2 = layer.visualize(image)
    # H = H / 3.1415926 + 0.5
    # print(H.min(), H.max())

    # torchvision.utils.save_image(H, f"H-{name}.jpg")
    # torchvision.utils.save_image(S, f"S-{name}.jpg", normalize=True)
    # torchvision.utils.save_image(Ww, f"Ww-{name}.jpg")
    # torchvision.utils.save_image(Wlw2, f"Wlw2-{name}.jpg")
    # torchvision.utils.save_image(Wllw2, f"Wllw2-{name}.jpg")


    # image = cv2.imread("demo.png")
    # image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # image_h = image_hsv[:,:,0:1]
    # image_s = image_hsv[:,:,1:2]
    # image_v = image_hsv[:,:,2:3]

    # cv2.imwrite("image_h.png", image_h)
    # cv2.imwrite("image_s.png", image_s)
    # cv2.imwrite("image_v.png", image_v)