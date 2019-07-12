import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import random
from . import unprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')


class loadImgs(data.Dataset):
    
    def __init__(self, args, image_in, loader=default_loader, mode='train'):
        self.image_in = image_in
        self.loader   = loader
        self.args     = args
        self.mode     = mode

    def align_to_64(self, img):
        pre_width   = img.size[0]
        pre_height  = img.size[1]
        new_width   = min(2048, int(pre_width /64)*64) # Capped at 2048
        new_height  = min(2048, int(pre_height/64)*64) # Capped at 2048
        center_crop_= transforms.Compose([transforms.CenterCrop((new_height, new_width))])
        img_aligned = center_crop_(img)
        return img_aligned

    def __getitem__(self, index):
        image_in  = self.image_in[index]

        if self.mode == 'train':
            image_in_img = self.loader(self.args.train_dir + '/' + image_in)
            if self.args.load_size is not None:
                load_size = self.args.load_size.strip('[]').split(', ')
                load_size = [int(item) for item in load_size]
                image_in_img = image_in_img.resize((load_size[1], load_size[0]))
            if self.args.crop_size is not None:
                w, h      = image_in_img.size
                crop_size = self.args.crop_size.strip('[]').split(', ')
                crop_size = [int(item) for item in crop_size]
                th, tw    = crop_size[0], crop_size[1]
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                image_in_img = image_in_img.crop((x1, y1, x1 + tw, y1 + th))
            hflip = random.random() < 0.5
            if hflip:
                image_in_img = transforms.functional.hflip(image_in_img)
            vflip = random.random() < 0.5
            if vflip:
                image_in_img = transforms.functional.vflip(image_in_img)
            rotate= random.random() < 0.5
            if rotate:
                degree = random.randint(-20, 20)
                image_in_img = transforms.functional.rotate(image_in_img, degree)
        elif self.mode == 'val' or self.mode == 'test':
            image_in_img = self.loader(self.args.test_dir + '/' + image_in)
            image_in_img = self.align_to_64(image_in_img)
        else:
            print('Unrecognized mode! Please select either "train" or "val"')
            raise NotImplementedError

        t_list = [transforms.ToTensor()]
        composed_transform    = transforms.Compose(t_list)
        image_in_img = composed_transform(image_in_img)

        # Adding unprocessing to raw here!
        image_in_img, metadata = unprocess.unprocess(image_in_img)
        shot_noise, read_noise = unprocess.random_noise_levels()
        noisy_img              = unprocess.add_noise(image_in_img, shot_noise, read_noise)
        # Approximation of variance is calculated using noisy image (rather than clean
        # image), since that is what will be avaiable during evaluation.
        variance               = shot_noise * noisy_img + read_noise
        inputs                 = {'noisy_img': noisy_img, 'variance': variance}
        inputs.update(metadata)
        labels = image_in_img
        return inputs, labels

    def __len__(self):
        return len(self.image_in)
