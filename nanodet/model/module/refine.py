import torch
import numpy as np
import torch.nn as nn
from nanodet.model.module.conv import ConvModule as Conv
from nanodet.model.module.conv import DepthwiseConvModule
from torchvision.ops import deform_conv2d as dcnv2


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class DCN_offset(nn.Module):
    """Compute the star deformable conv offsets.

    Args:
        bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
        gradient_mul (float): Gradient multiplier.
        stride (int): The corresponding stride for feature maps,
            used to project the bbox onto the feature map.

    Returns:
        dcn_offsets (Tensor): The offsets for deformable convolution.
    """
    def __init__(self,num_dconv_points):
        super().__init__()
        self.num_dconv_points = num_dconv_points
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)


    def forward(self,bbox_pred,stride,gradient_mul=0.1):
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2  # x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset

class DeformConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=3,stride = 1,padding = 0,groups=1):
        super().__init__()
        self.kh, self.kw = kernel, kernel
        self.stride = stride
        self.padding = padding
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.convoff = DepthwiseConvModule(self.in_channel,2*self.kh*self.kw,3,1,1)
        self.conmask = DepthwiseConvModule(self.in_channel,self.kh*self.kw,3,1,1)
        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=self.kh, stride=self.stride, padding=self.padding)
    def forward(self,x,offset):
        # b,c,h,w = x.shape
        # h_out=(h-self.kw+2*self.padding)/self.stride+1

        offset = offset
        mask = torch.sigmoid(self.conmask(x))

        out = dcnv2(input=x, offset=offset, weight=self.conv.weight, mask=mask, padding=(1, 1))
        return out

class Refinebox(nn.Module):
    def __init__(self, in_channel, stride ,reg_max):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dcn_offset = DCN_offset(num_dconv_points=9)
        self.stride = stride
        self.scale = Scale(1.0)
        self.relu = nn.ReLU(inplace=True)
        self.reg_refine_dconv =  DeformConv2d(in_channel,in_channel,3,1,padding=1)
        self.reg_preds = DepthwiseConvModule(in_channel, 4, 1, padding=0)

    def forward(self,pre_box,reg_feat):
        pre_box = self.scale(pre_box).float().exp() * self.stride
        dcn_offset = self.dcn_offset(pre_box,stride=self.stride).to(reg_feat.dtype)
        reg_feat_ref = self.relu(self.reg_refine_dconv(reg_feat, dcn_offset))
        bbox_pred_refine = self.scale(self.reg_preds(reg_feat_ref)).float().exp()
        bbox_pred_refine = bbox_pred_refine * pre_box.detach()
        return pre_box,bbox_pred_refine

