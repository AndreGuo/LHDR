import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


### Architecture utilities ###
def which_act(act_type):
    if act_type == 'relu':
        act = nn.ReLU(inplace=True)
    elif act_type == 'relu6':
        act = nn.ReLU6(inplace=True)
    elif act_type == 'leakyrelu':
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'elu':
        act = nn.ELU()
    else:
        raise AttributeError('Unsupported activation_type!')
    return act


def duplicate_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class BrightMask(nn.Module):
    def __init__(self, nf=3, threshold=0.9, need_normalize=False, rest_valid=False):
        super(BrightMask, self).__init__()
        self.t = threshold
        self.add_norm = nn.InstanceNorm2d(nf, affine=True) if need_normalize else None
        self.rv = rest_valid

    def forward(self, x):
        x = self.add_norm(x) if self.add_norm is not None else x
        y = (x - 1)/(self.t - 1) if self.rv else (x - self.t)/(1 - self.t)
        # map [t,1] to [0,1]; [0,t] to 0 when rest_valid, else map [t,1] to [1,0]; [0,r] to 1
        return torch.clamp(y, min=0, max=1)


class QuadGroupConvMultiK(nn.Module):
    def __init__(self, nf=32):
        super(QuadGroupConvMultiK, self).__init__()
        assert nf % 4 == 0, ''

        self.group_nf = nf // 4
        self.conv_group_1 = nn.Conv2d(nf // 4, nf // 4, 1, 1, 0)
        self.conv_group_2 = nn.Conv2d(nf // 4, nf // 4, 3, 1, 1)
        self.conv_group_3 = nn.Conv2d(nf // 4, nf // 4, 5, 1, 2)
        self.conv_group_4 = nn.Conv2d(nf // 4, nf // 4, 7, 1, 3)

    def forward(self, x):
        x = torch.split(x, self.group_nf, dim=1)
        y1 = self.conv_group_1(x[0])
        y2 = self.conv_group_2(x[1])
        y3 = self.conv_group_3(x[2])
        y4 = self.conv_group_4(x[3])
        return torch.cat([y1, y2, y3, y4], dim=1)


class Channel_Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32, act_type='leakyrelu'):
        super(Channel_Condition, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, nf//2, 3, 2, 1)
        self.conv2 = QuadGroupConvMultiK(nf=nf//2)
        self.conv3 = nn.Conv2d(nf//2, nf, 3, 2, 1)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        self.act = which_act(act_type)

    def forward(self, x):
        conv1_out = self.act(self.conv1(x))
        conv2_out = self.act(self.conv2(conv1_out))
        conv3_out = self.act(self.conv3(conv2_out))
        # out = self.pool(conv3_out).squeeze(2).squeeze(2)
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)  # [n, c]
        return out


# ECCV18 Partial Convolution, from github.com/NVIDIA/partialconv, some error fixed
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'multi_channel' in kwargs:  # whether the mask is multi-channel
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.maskUpdater.shape[1] * self.maskUpdater.shape[2] * self.maskUpdater.shape[3]
        self.last_size = [None, None, None, None]
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, x, mask_in=None):
        # Input: (content, mask)
        assert len(x.shape) == 4, 'Please check tensor shape!'
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with torch.no_grad():
                if self.maskUpdater.type() != x.type():
                    self.maskUpdater = self.maskUpdater.to(x)

                mask = mask_in
                if mask_in is None:
                    # if mask is not provided, create one
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0], x.data.shape[1], x.data.shape[2], x.data.shape[3]).to(x)
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2], x.data.shape[3]).to(x)

                self.update_mask = F.conv2d(mask, self.maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)  # 1e-6 for mixed precision training
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(x, mask) if mask_in is not None else x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class ResBlock2ConvPConv(nn.Module):
    def __init__(self, nf=32, act_type='leakyrelu'):
        super(ResBlock2ConvPConv, self).__init__()
        self.act = which_act(act_type)
        self.pconv_1 = PartialConv2d(nf, nf, 3, 1, 1, groups=4, bias=True, multi_channel=True, return_mask=True)
        self.pconv_2 = PartialConv2d(nf, nf, 3, 1, 1, bias=True, multi_channel=True, return_mask=True)

    def forward(self, x):
        # x[0]: fea; x[1]: cond_mask (binary)
        fea, new_mask = self.pconv_1(x[0], x[1])
        fea = self.act(fea)
        fea, new_mask = self.pconv_2(fea, new_mask)
        fea = self.act(fea)
        return fea + x[0], new_mask


class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=32, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv1 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv2 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv2 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):
        # we assume x[0]: content; x[1]: s_cond
        scale = self.SFT_scale_conv2(F.leaky_relu(self.SFT_scale_conv1(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv2(F.leaky_relu(self.SFT_shift_conv1(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock2ConvSFT(nn.Module):
    def __init__(self, nf=32, act_type='leakyrelu'):
        super(ResBlock2ConvSFT, self).__init__()

        self.act = which_act(act_type)

        # self.conv1 = QuadGroupConvMultiK(nf=nf)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, groups=4)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.sft1 = SFTLayer(in_nc=nf, out_nc=nf, nf=nf)
        self.sft2 = SFTLayer(in_nc=nf, out_nc=nf, nf=nf)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft1(x)
        fea = self.act(self.conv1(fea))
        fea = self.sft2((fea, x[1]))
        fea = self.act(self.conv2(fea))
        return fea + x[0], x[1]


class OctConvFirst(nn.Module):
    def __init__(self, in_nc, out_nc, k_size, alpha=0.5, act_type='leakyrelu'):
        super(OctConvFirst, self).__init__()

        out_nc_l = int(out_nc * alpha)
        out_nc_h = out_nc - out_nc_l
        pad = (k_size - 1) // 2

        self.conv_h = nn.Conv2d(in_nc, out_nc_h, k_size, 1, pad)
        self.conv_l = nn.Conv2d(in_nc, out_nc_l, k_size, 2, pad)

        self.act = which_act(act_type)

    def forward(self, x):
        # x is not in [h, l]
        out_h = self.act(self.conv_h(x))
        out_l = self.act(self.conv_l(x))
        return out_h, out_l


class ResDenseBlock(nn.Module):
    def __init__(self, nf=32, act_type='leakyrelu'):
        super(ResDenseBlock, self).__init__()

        self.act = which_act(act_type)

        self.conv_1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_5 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        y1 = self.act(self.conv_1(x))
        y2 = self.act(self.conv_2(x + y1))
        y3 = self.act(self.conv_3(x + y1 + y2))
        y4 = self.act(self.conv_4(x + y1 + y2 + y3))
        y5 = self.conv_5(x + y1 + y2 + y3 + y4)
        return y5 + x


### Step 1 ###
class MLPNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, act_type='leakyrelu'):
        super(MLPNet, self).__init__()

        self.nf = nf
        self.out_nc = out_nc

        self.conv1 = nn.Conv2d(in_nc, nf, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf*2, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf*2, nf, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf, out_nc, 1, 1, bias=True)

        self.act = which_act(act_type)

        self.cond_net = Channel_Condition(in_nc=in_nc, nf=nf,  act_type=act_type)

        self.cond_scale1 = nn.Linear(nf, nf, bias=True)
        self.cond_scale2 = nn.Linear(nf, nf*2, bias=True)
        self.cond_scale3 = nn.Linear(nf, nf, bias=True)
        self.cond_scale4 = nn.Linear(nf, out_nc, bias=True)

        self.cond_shift1 = nn.Linear(nf, nf, bias=True)
        self.cond_shift2 = nn.Linear(nf, nf*2, bias=True)
        self.cond_shift3 = nn.Linear(nf, nf, bias=True)
        self.cond_shift4 = nn.Linear(nf, out_nc, bias=True)

    def forward(self, x):
        # x: [img, s_cond(unused), c_cond], form dataloader
        cond = self.cond_net(x[2])

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        scale4 = self.cond_scale4(cond)
        shift4 = self.cond_shift4(cond)

        out = self.conv1(x[0])
        out = self.act(out * scale1.view(-1, self.nf, 1, 1) + shift1.view(-1, self.nf, 1, 1) + out)

        out = self.conv2(out)
        out = self.act(out * scale2.view(-1, self.nf*2, 1, 1) + shift2.view(-1, self.nf*2, 1, 1) + out)

        out = self.conv3(out)
        out = self.act(out * scale3.view(-1, self.nf, 1, 1) + shift3.view(-1, self.nf, 1, 1) + out)

        out = self.conv4(out)
        out = self.act(out * scale4.view(-1, self.out_nc, 1, 1) + shift4.view(-1, self.out_nc, 1, 1) + out)
        return out


### UNet in step 2 ###
class GlobalUNetPConvSFT(nn.Module):
    def __init__(self, in_nc=3, nf=32, act_type='leakyrelu'):
        super(GlobalUNetPConvSFT, self).__init__()

        self.act = which_act(act_type)

        self.mask_est_pconv = nn.Sequential(BrightMask(nf=in_nc, threshold=0.9, rest_valid=True),
                                            nn.Conv2d(in_nc, nf, 1))
        self.mask_upscale = nn.UpsamplingNearest2d(scale_factor=2)
        self.mask_downscale = nn.AvgPool2d(3, 2, 1)

        # Encoder w. PConv, Decoder w. SFT
        # UNet main branch, input: 1/2 scale
        RB_SFT = functools.partial(ResBlock2ConvSFT, nf=nf, act_type=act_type)
        RB_PConv = functools.partial(ResBlock2ConvPConv, nf=nf, act_type=act_type)
        self.recon_trunk_f = duplicate_layer(RB_PConv, 2)
        self.down_conv = nn.Conv2d(nf, nf, 3, 2, 1)
        self.recon_trunk_mp = duplicate_layer(RB_PConv, 2)
        self.recon_trunk_ms = duplicate_layer(RB_SFT, 2)
        self.up_conv = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.recon_trunk_e = duplicate_layer(RB_SFT, 2)

        # Spacial condition branch
        self.cond_com = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), self.act,
                                      nn.Conv2d(nf, nf, 3, 1, 1, groups=4), self.act)
        self.cond_1e = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1, groups=4), self.act, nn.Conv2d(nf, nf, 3, 1, 1))
        self.cond_2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1, groups=4), self.act, nn.Conv2d(nf, nf, 3, 2, 1))

    def forward(self, x):
        # this step is not 1st, data is not from data loader,
        # make sure x[0]: content; x[1]: spacial_cond_img

        mask_p = self.mask_downscale(self.mask_est_pconv(x[1]))
        mask_s = x[1]  # now masked outside the network

        cond_com = self.cond_com(mask_s)
        cond_1e = self.cond_1e(cond_com)
        cond_2 = self.cond_2(cond_com)

        fea_1f, mask_p = self.recon_trunk_f((x[0], mask_p))
        fea_2f, mask_p = self.act(self.down_conv(fea_1f)), self.mask_downscale(mask_p)
        fea_2e, _ = self.recon_trunk_mp((fea_2f, mask_p))
        fea_2e, _ = self.recon_trunk_ms((fea_2e, cond_2))
        fea_2e = fea_2e + fea_2f
        fea_1e = self.act(self.up_conv(fea_2e))
        fea_1e, _ = self.recon_trunk_e((fea_1e, cond_1e))
        fea_1e = fea_1e + fea_1f

        return fea_1e + x[0]


### Step 2 ###
class MultiScaleNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, act_type='leakyrelu'):
        super(MultiScaleNet, self).__init__()

        self.act = which_act(act_type)

        # decompose
        self.decompose = OctConvFirst(in_nc, nf*2, 3, alpha=0.5, act_type=act_type)

        # local branch
        self.local_branch = ResDenseBlock(nf=nf, act_type=act_type)
        self.conv0 = nn.Conv2d(nf, out_nc, 3, 1, 1)

        # global branch
        self.global_branch = GlobalUNetPConvSFT(in_nc=in_nc, nf=nf, act_type=act_type)
        self.upconv0 = nn.Sequential(nn.Conv2d(nf, out_nc*4, 3, 1, 1), nn.PixelShuffle(2))

    def forward(self, x):
        # this step is not 1st, data is not from data loader,
        # make sure x[0]: content; x[1]: spacial_cond_img

        detail, base = self.decompose(x[0])

        detail = self.local_branch(detail)
        detail = self.act(self.conv0(detail))

        base = self.global_branch((base, x[1]))
        base = self.act(self.upconv0(base))

        return F.relu(base + detail)


### Step 1 & Step 2 ###
class LiteHDRNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, act_type='leakyrelu'):
        super(LiteHDRNet, self).__init__()

        self.mlp = MLPNet(in_nc=in_nc, out_nc=out_nc, nf=nf, act_type=act_type)
        self.msnet = MultiScaleNet(in_nc=out_nc, out_nc=out_nc, nf=nf,act_type=act_type)

    def forward(self, x):
        # x: [img, s_cond, c_cond]
        out1 = self.mlp(x)
        out2 = self.msnet((out1, x[1]))
        return out2
