import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from model import AVENet
from torchvision.utils import save_image,make_grid
from inference.generate_images import get_model
from scipy import signal
import soundfile as sf
import librosa
from pydub import AudioSegment
from torchvision.transforms.functional import to_pil_image
import pdb
import shutil


def get_arguments():
    parser = argparse.ArgumentParser()
    #Definine Sound2Scene model (audio encoder)
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=2048,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--checkpoint_vggish', dest='checkpoint',help='Path of checkpoint file for load model')

    #Defining image encoder and image decoder
    parser.add_argument('--root_path', default="./checkpoints")
    parser.add_argument('--model', default="icgan")
    parser.add_argument('--model_backbone', default="biggan")
    parser.add_argument('--resolution', default=128)
    parser.add_argument("--z_var", type=float, default=1, help="Noise variance: %(default)s)")
    parser.add_argument(
        "--trained_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset in which the model has been trained on.",
    )

    #Defining directories
    parser.add_argument("--dataset", type=str, default="vgg", choices=["vgg", "vegas"],help="Dataset in which the model has been trained on.", )
    parser.add_argument("--wav_path", type=str)
    parser.add_argument("--out_path", type=str)



    return parser.parse_args()

def load_data(args, batch_size=None):
    if batch_size==None:
        batch_size=args.batch_size

    if args.dataset == 'vgg':
        aud_path = os.path.join(args.vgg_dir, "audios")
        vid_path = os.path.join(args.vgg_dir, "frames_10fps")
        train_dataset = GetVGGSound(args.data_txt, args.annotation, aud_path, vid_path, vid_path)
        test_dataset = GetVGGSound(args.data_t_txt, args.annotation, aud_path, vid_path, vid_path)

    elif args.dataset == "vegas":
        train_dataset = GetAudioVideoDataset(args.data_txt, args.aud_path, args.emb_path, args.img_path)
        test_dataset = GetAudioVideoDataset(args.data_t_txt, args.aud_t_path, args.emb_t_path, args.img_t_path)

    #train_loader=None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def showImage(args,generator,feature_extractor, emb):
    # define noise_vectors
    z = torch.empty(8,generator.dim_z).normal_(mean=0, std=args.z_var)

    #normalize audio embedding
    emb /= torch.linalg.norm(emb, dim=-1, keepdims=True)
    emb_ = torch.tile(emb, (8,1))

    gen_img = generator(z.cuda(), None, emb_)
    gen_img = torch.clamp(gen_img,-1., 1.)

    output = make_grid(gen_img, normalize=True, scale_each=True, nrow=8)

    return output

def gen_name(tuple):
    name = ''
    for i in tuple:
        name = name+i+"_"
    return name

def audio2spectrogra(wav_file):
    samples, samplerate = sf.read(wav_file)

    # repeat in case audio is too short
    resamples = np.tile(samples, 10)[:160000]
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    spectrogram =  torch.from_numpy(spectrogram).unsqueeze(0)
    return spectrogram

def generate_images(args, model, generator,feature_extractor,device):
    audio_paths = args.wav_path
    save_path = args.out_path
    os.makedirs(save_path, exist_ok=True)

    for audio in os.listdir(audio_paths):
        audio = os.path.join(audio_paths,audio)
        #audio = "./samples/inference/chainsaw.wav"

        spectrogram = audio2spectrogra(audio)
        spectrogram = Variable(spectrogram).to(device)
        _,emb = model(spectrogram.unsqueeze(1).float())
        output = showImage(args, generator, feature_extractor, emb)

        save_name = audio.split("/")[-1].split(".")[0]

        save_final = os.path.join(save_path, save_name+".png")

        save_image(output.cpu(), save_final)



def main():
    import random
    random_seed=1234
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load sound2scene model
    checkpoint = torch.load(args.checkpoint,map_location=device)
    model = AVENet(args).to(device)
    model.load_state_dict(checkpoint)
    model.eval()


    #load image decoder
    suffix = (
        "_nofeataug"
        if args.resolution == 256
        and args.trained_dataset == "imagenet"
        else ""
    )
    exp_name = "%s_%s_%s_res%i%s" % (
        args.model,
        args.model_backbone,
        args.trained_dataset,
        args.resolution,
        suffix,
    )
    generator, feature_extractor = get_model(
        exp_name, args.root_path, args.model_backbone, device=device
    )

    generate_images(args,model, generator,feature_extractor,device)

if __name__=='__main__':
    main()




