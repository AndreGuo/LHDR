import os
import time
from os import path
import argparse
import numpy as np
import torch
import cv2
from network import LiteHDRNet


### System utilities ###
def process_path(directory, create=False):
    directory = path.expanduser(directory)
    directory = path.normpath(directory)
    directory = path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = path.splitext(path.basename(directory))
    return path.dirname(directory), name, ext


def compose(transforms):
    """Composes list of transforms (each accept and return one item)"""
    assert isinstance(transforms, list)
    for transform in transforms:
        assert callable(transform), "list of functions expected"

    def composition(obj):
        """Composite function"""
        for transform in transforms:
            obj = transform(obj)
        return obj
    return composition


def str2bool(x):
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True


def create_name(inp, tag, ext, out, extra_tag):
    root, name, _ = split_path(inp)
    if extra_tag is not None:
        tag = '{0}_{1}'.format(tag, extra_tag)
    if out is not None:
        root = out
    return path.join(root, '{0}_{1}.{2}'.format(name, tag, ext))


### Image utilities ###
def np2torch(img):
    img = img[:, :, [2, 1, 0]]
    return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
# def np2torch(np_img):
#     rgb = np_img[:, :, (2, 1, 0)]
#     return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


def torch2np(t_img):
    img_np = t_img.detach().numpy()
    return np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)).astype(np.float32)
# def torch2np(t_img):
#     return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]


def resize(x):
    return cv2.resize(x, (opt.width, opt.height)) if opt.resize else x


class Exposure(object):
    def __init__(self, stops, gamma):
        self.stops = stops
        self.gamma = gamma

    def process(self, img):
        return np.clip(img*(2**self.stops), 0, 1)**self.gamma


### Parameters ###
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('ldr', nargs='+', type=process_path, help='Ldr image(s)')
arg('-out', type=lambda x: process_path(x, True), default=None, help='Output location.')
arg('-resize', type=str2bool, default=False, help='Use resized input.')
arg('-width', type=int, default=1920, help='Image width resizing.')
arg('-height', type=int, default=1080, help='Image height resizing.')
arg('-tag', default=None, help='Tag for outputs.')
arg('-use_gpu', type=str2bool, default=torch.cuda.is_available(), help='Use GPU for prediction.')
arg('-out_format', choices=['hdr', 'exr', 'png'], default='hdr', help='Encapsulation of output HDR image.')
arg('-stops', type=float, default=0.0, help='Stops (loosely defined here) for exposure tone mapping.')
arg('-gamma', type=float, default=1.0, help='Gamma curve value (if tone mapping).')
arg('-linearize', type=str2bool, default=True, help='Linearize the output HDR.')
arg('-half', type=str2bool, default=False, help='Change the precision of both param. & network to half float.')
opt = parser.parse_args()


### Load network ###
net = LiteHDRNet(in_nc=3, out_nc=3, nf=32, act_type='leakyrelu')
net.load_state_dict(torch.load('params.pth', map_location=lambda s, l: s))


### Loading images ###
preprocess = compose([lambda x: x.astype('float32'), resize])

for ldr_file in opt.ldr:
    loaded = cv2.imread(ldr_file, flags=cv2.IMREAD_UNCHANGED) / 255.0
    print('Could not load {0}'.format(ldr_file)) if loaded is None else print('Image {0} loaded!'.format(ldr_file))
    start = time.time()
    ldr_input = preprocess(loaded)

    # copy input numpy to [img, s_cond, c_cond] to suit the network model
    s_cond_prior = ldr_input.copy()
    s_cond_prior = np.clip((s_cond_prior - 0.9)/(1 - 0.9), 0, 1)  # now masked outside the network
    c_cond_prior = cv2.resize(ldr_input.copy(), (0, 0), fx=0.25, fy=0.25)
        
    ldr_input_t = np2torch(ldr_input).unsqueeze(dim=0)
    s_cond_prior_t = np2torch(s_cond_prior).unsqueeze(dim=0)
    c_cond_prior_t = np2torch(c_cond_prior).unsqueeze(dim=0)

    if opt.use_gpu:
        net.cuda()
        ldr_input_t = ldr_input_t.cuda()
        s_cond_prior_t = s_cond_prior_t.cuda()
        c_cond_prior_t = c_cond_prior_t.cuda()

    if opt.half:
        net.half()
        ldr_input_t = ldr_input_t.half()
        s_cond_prior_t = s_cond_prior_t.half()
        c_cond_prior_t = c_cond_prior_t.half()

    x = (ldr_input_t, s_cond_prior_t, c_cond_prior_t)
    prediction = net(x)
    prediction = prediction.detach()[0].float().cpu()
    prediction = torch2np(prediction)

    prediction = prediction / prediction.max()

    if opt.linearize:
        prediction = prediction ** (1 / 0.45)

    if opt.out_format == 'hdr':
        out_name = create_name(ldr_file, 'prediction', 'hdr', opt.out, opt.tag)
        cv2.imwrite(out_name, prediction)
    elif opt.out_format == 'exr':
        raise AttributeError('Unsupported output format, you can install additional package openEXR to support it')
    elif opt.out_format == 'png':
        out_name = create_name(ldr_file, 'prediction', 'png', opt.out, opt.tag)
        prediction = np.round(prediction * 65535.0).astype(np.uint16)
        cv2.imwrite(out_name, prediction)
    else:
        raise AttributeError('Unsupported output format!')

    end = time.time()
    print('Finish processing {0}. \n takes {1} seconds. \n -------------------------------------'
          ''.format(ldr_file, '%.04f' % (end-start)))

