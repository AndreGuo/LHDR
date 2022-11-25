# LHDR_PyTorch
The official PyTorch implementation of paper
"***LHDR: HDR Reconstructionfor Legacy Content using a Lightweight DNN***"
([paper (ArXiv)](https://arxiv.org/abs/2211.11270),
[paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Guo_LHDR_HDR_Reconstruction_for_Legacy_Content_using_a_Lightweight_DNN_ACCV_2022_paper.pdf),
[supplementary material](https://openaccess.thecvf.com/content/ACCV2022/supplemental/Guo_LHDR_HDR_Reconstruction_ACCV_2022_supplemental.pdf))
in ACCV2022.
    
    @inproceedings{guo2022lhdr,
      title={LHDR: HDR Reconstructionfor Legacy Content using a Lightweight DNN}, 
      author={Guo, Cheng and Jiang Xiuhua},
      booktitle={Proceedings of the 16th Asian Conference on Computer Vision},
      year={2022},
      pages={3155-3171}
    }

---

## 1. Overview

### 1.1 Our scope

This method belongs to ```computer vision (CV)```
 -> ```image-to-image (I2I)``` -> ```low-level vision```
 -> ```high dynamic range (HDR) related```.

**NOTE THAT** this method DONNOT belong to ```multi-exposure HDR imaging``` which
takes merge multiple images into HDR, it belongs to ```single-image reconstruction (SI-HDR)```
who predict HDR from single image.

### 1.2 Key features

Our ***LHDR*** tackles 2 issues of current methods:

- The first ***L***: current methods could not well process ***L***egacy content with degradations *e.g.*
camera noise, JPEG compression and over-exposure *etc.*
- The second ***L***: current methods are too bulky for real application,
so we have to design a more ***L***ightweight network.

---

## 2. Getting Started

### 2.1 Prerequisites

- Python
- PyTorch
- OpenCV
- NumPy
- NVIDIA GPU, cnDNN, CUDA (only for training)

### 2.2 How to test

Run `test.py` with below configuration(s):

```bash
python test.py imgName.jpg
```

When batch processing, use wildcard `*`:

```bash
python test.py imgPath/*.jpg
```

Add below configuration(s) for specific propose:

| Propose                                                      |                                    Configuration                                     |
|:-------------------------------------------------------------|:------------------------------------------------------------------------------------:|
| Specify output path                                          |                                  `-out resultDir/`                                   |
| Resize image before inference                                |                       `-resize True -height newH -width newW`                        |
| Add filename tag                                             |                                    `-tag yourTag`                                    |
| Force CPU processing                                         |                                   `-use_gpu False`                                   |
| Change precision to half (when GRAM OMM)                     |                                     `-half True`                                     |
| Save result in another format (defalut `.hdr`)               | `-out_format suffix`<br>`png` as 16bit .png<br>`exr` require extra package `openEXR` |
| Donnot linerize the output HDR                               |                                  `-linearize False`                                  |



### 2.3 How to train

Put our ```network.py``` under your own PyTorch scheme, *e.g.* [BasicSR](https://github.com/xinntao/BasicSR), 
and change the dataloader *etc.*, remember to:

- Normalize input SDR (*img*) by dividing 255.0.
Note that the input to network is a tuple [*img*,*s_cond*,*c_cond*],
see ```line145``` in ```test.py``` for detail.
- Transfer the normalized label HDR by to non-linear domain
by 0.45 exponent (our pre-processing).
- Change channel order BGR (if so) to RGB.

## Contact

Guo Cheng ([Andre Guo](https://orcid.org/orcid=0000-0002-2660-2267)) guocheng@cuc.edu.cn

- *State Key Laboratory of Media Convergence and Communication (MCC),
Communication University of China (CUC), Beijing, China.*
- *Peng Cheng Laboratory (PCL), Shenzhen, China.*
