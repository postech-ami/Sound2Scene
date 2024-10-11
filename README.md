# Sound2Scene (CVPR 2023)

### [Project Page](https://sound2scene.github.io/) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Sung-Bin_Sound_to_Visual_Scene_Generation_by_Audio-to-Visual_Latent_Alignment_CVPR_2023_paper.html)
This repository contains a pytorch implementation for the CVPR 2023 paper, [Sound2Scene: Sound to visual scene generation by audio-to-visual latent alignment](https://openaccess.thecvf.com/content/CVPR2023/html/Sung-Bin_Sound_to_Visual_Scene_Generation_by_Audio-to-Visual_Latent_Alignment_CVPR_2023_paper.html). Sound2Scene is a sound-to-image generative model which is trained solely from unlabeled videos to generate images from sound.<br><br>

![teaser1](https://github.com/postech-ami/Sound2Scene/assets/59387731/9c1a2d37-38e0-4525-9dc2-74002ee4c2e2)

## Getting started
This code was developed on Ubuntu 18.04 with Python 3.8, CUDA 11.1 and PyTorch 1.8.0. Later versions should work, but have not been tested.

### Installation 
Create and activate a virtual environment to work in.
```
conda create --name sound2scene python=3.8.8
conda activate sound2scene
```

Install [PyTorch](https://pytorch.org/). For CUDA 11.1, this would look like:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
```

### Download Models
To run Sound2Scene, you need to download image encoder (SWAV), image decoder (BigGAN) and Sound2Scene model.
Download [Sound2Scene](https://drive.google.com/file/d/1MfQo9Y6cBwSo9sYkwj2gG9kNa_1fuaUJ/view?usp=sharing) | [SWAV](https://drive.google.com/file/d/1_DjU6MBZwQTQzNdlktr12eUZszaRvPX5/view?usp=sharing) | [BigGAN](https://drive.google.com/drive/folders/1nlpQ-D2zQNlEWDOKidOV-p4Ny26KHvlb?usp=sharing).

After downloading the models, place them in ./checkpoints.
```
./checkpoints/icgan_biggan_imagenet_res128
./checkpoints/sound2scene.pth
./checkpoints/swav.pth.tar
```

### Highly correlated audio-visual pair dataset
We provide the annotations of the highly correlated audio-visual pairs from the VGGSound dataset.

Download [top1_boxes_top10_moments.json](https://drive.google.com/file/d/1uFht0YV8al9RqMPR2Umn99xWPluOU-UQ/view?usp=drive_link)

The annotation file contains each video name with the corresponding top 10 audio-visually correlated frame numbers.

```
{'9fhhMaXTraI_44000_54000': [47, 46, 45, 23, 42, 9, 44, 56, 27, 17],
'G_JwMzRLRNo_252000_262000': [2, 1, 26, 29, 15, 16, 11, 3, 14, 23], ...}

# 9fhhMaXTraI_44000_54000: video name
# [47, 46, 45, 23, 42, 9, 44, 56, 27, 17]: frame number (e.g., 47th, 46th frame, ...)
# 47th frame is the highest audio-visually correlated frame
```

Please follow the steps below to select a highly correlated audio-visual pair dataset.

**(Step 1)** Download the training dataset from [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

**(Step 2)** Extract the frames of each video in 10 fps.

**(Step 3)** Select the frame that is mentioned in the annotation file.

If you find this dataset helpful, please consider also citing: 
[Less Can Be More: Sound Source Localization With a Classification Model](https://openaccess.thecvf.com/content/WACV2022/html/Senocak_Less_Can_Be_More_Sound_Source_Localization_With_a_Classification_WACV_2022_paper.html).

The VEGAS dataset is available [here](https://drive.google.com/file/d/1ah2s3m96Nz0MUQX9Z9i4pRsqH89rTtu4/view?usp=drive_link).

## Training Sound2Scene
Run below command to train the model.

We provide sample image and audio pairs in **./samples/training**.

The samples are for checking the training code.

For the full dataset, please download the training dataset from [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) or VEGAS.

Although we provide the categories which we used ([category_list](https://github.com/postech-ami/Sound2Scene/blob/main/samples/categories.txt)), no category information were used for training.
```
python train.py --data_path [path containing image and audio pairs] --save_path [path for saving the checkpoints]

#or

bash train.sh
```

## Evaluating Sound2Scene
(1) We used off-the-shelf CLIP model (``Vit-B/32'') to evaluate R@k performance.

(2) We trained the [Inception model](https://drive.google.com/file/d/1GbZ25SShTssQ-G5Ynhzsjwz6QkPWWQNm/view?usp=drive_link) on VGGSound for measuring FID and Inception score.

## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@inproceeding{sung2023sound,
  author    = {Sung-Bin, Kim and Senocak, Arda and Ha, Hyunwoo and Owens, Andrew and Oh, Tae-Hyun},
  title     = {Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment},
  booktitle   = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
````

## Acknowledgment
This work was supported by IITP grant funded by Korea government (MSIT) (No.2021-0-02068, Artificial Intelligence Innovation Hub; No.2022-0-00124, Development of Artificial Intelligence Technology for Self-Improving Competency-Aware Learning Capabilities). The GPU resource was supported by the HPC Support Project, MSIT and NIPA.

The implementation of Sound2Scene borrows the most of the codebases from the seminal prior work, [ICGAN](https://github.com/facebookresearch/ic_gan) and [VGGSound](https://github.com/hche11/VGGSound).
We thank the authors of both work who made their code public. Also If you find our work helpful, please consider citing them as well.


