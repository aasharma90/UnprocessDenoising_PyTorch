# Denoises RAW iamges from the Darmstadt dataset.

import os
import h5py
import numpy as np
import torch
from torchvision import transforms
from torchvision import utils
from dataloader import process
from models import *
import argparse


def get_arguments():
  """Parse all the arguments provided from the CLI.
  Returns:
  A list of parsed arguments.
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--load_model", type=str, required=True, default=None,
                      help="Location from which any pre-trained model needs to be loaded.")
  parser.add_argument("--data_dir", type=str, required=True, default=None,
                      help="Directory containing the Darmstadt RAW images.")
  parser.add_argument("--results_dir", type=str, required=True, default=None,
                      help="Directory to store the results in.")
  parser.add_argument('--gpu_id', type=int, default=0,
                      help='Select the args.gpu_id to run the code on')
  return parser.parse_args()


if __name__ == '__main__':
  """Denoises all bounding boxes in all raw images from the DND dataset.

  The resulting denoised images are saved to disk.

  Args:
    denoiser: Function handle called as:
        denoised_img = denoiser(noisy_img, shot_noise, read_noise).
    data_dir: Folder where the DND dataset resides
    output_dir: Folder where denoised output should be written to

  Returns:
    None
  """

  # Gets arguments
  args = get_arguments()

  # Creates the results directory is not existing already
  if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

  # Loads image information and bounding boxes.
  info = h5py.File(os.path.join(args.data_dir, 'info.mat'), 'r')['info']
  bb = info['boundingboxes']

  # Create generator model
  args = get_arguments()
  torch.cuda.set_device(args.gpu_id)
  model = Generator().cuda()
  model = nn.DataParallel(model, device_ids=[args.gpu_id]).cuda()
  if args.load_model is not None:
    print('Loading pre-trained checkpoint %s' % args.load_model)
    model_psnr = torch.load(args.load_model)['avg_psnr']
    model_ssim = torch.load(args.load_model)['avg_ssim']
    print('Avg. PSNR and SSIM values recorded from the checkpoint: %f, %f' % (model_psnr, model_ssim))
    model_state_dict = torch.load(args.load_model)['state_dict']
    model.load_state_dict(model_state_dict)
      
  # Denoise each image.
  for i in range(0, 50):
    # Loads the noisy image.
    filename = os.path.join(args.data_dir, 'images_raw', '%04d.mat' % (i + 1))
    print('Processing file: %s' % filename)
    img = h5py.File(filename, 'r')
    noisy =np.float32(np.array(img['Inoisy']).T)

    # Loads raw Bayer color pattern.
    bayer_pattern = np.asarray(info[info['camera'][0][i]]['pattern']).tolist()
    # Load the camera's (or image's) ColorMatrix2 
    xyz2cam = torch.FloatTensor(np.reshape(np.asarray(info[info['camera'][0][i]]['ColorMatrix2']), (3, 3)))
    # print(bayer_pattern, xyz2cam)
    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    cam2rgb = torch.inverse(rgb2cam)
    # print(cam2rgb, cam2rgb.size())
    # Specify red and blue gains here (for White Balancing)
    asshotneutral = info[info['camera'][0][i]]['AsShotNeutral']
    # print(asshotneutral[1]/asshotneutral[0], asshotneutral[1]/asshotneutral[2])
    red_gain  =  torch.FloatTensor(asshotneutral[1]/asshotneutral[0])
    blue_gain =  torch.FloatTensor(asshotneutral[1]/asshotneutral[2])

    # Denoises each bounding box in this image.
    boxes = np.array(info[bb[0][i]]).T
    for k in range(20):
      # Crops the image to this bounding box.
      idx = [
          int(boxes[k, 0] - 1),
          int(boxes[k, 2]),
          int(boxes[k, 1] - 1),
          int(boxes[k, 3])
      ]
      noisy_crop = noisy[idx[0]:idx[1], idx[2]:idx[3]].copy()

      # Flips the raw image to ensure RGGB Bayer color pattern.
      if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
      elif (bayer_pattern == [[2, 1], [3, 2]]):
        noisy_crop = np.fliplr(noisy_crop)
      elif (bayer_pattern == [[2, 3], [1, 2]]):
        noisy_crop = np.flipud(noisy_crop)
      else:
        print('Warning: assuming unknown Bayer pattern is RGGB.')

      # Loads shot and read noise factors.
      nlf_h5 = info[info['nlf'][0][i]]
      shot_noise = nlf_h5['a'][0][0]
      read_noise = nlf_h5['b'][0][0]

      # Extracts each Bayer image plane.
      denoised_crop = noisy_crop.copy()
      height, width = noisy_crop.shape
      noisy_bayer   = []
      for yy in range(2):
        for xx in range(2):
          noisy_crop_c = noisy_crop[yy:height:2, xx:width:2].copy()
          noisy_bayer.append(noisy_crop_c)
      noisy_bayer = np.stack(noisy_bayer, axis=-1)
      # print(np.shape(noisy_bayer))
      variance    = shot_noise * noisy_bayer + read_noise

      totensor_   = transforms.ToTensor()
      noisy_bayer = torch.unsqueeze(totensor_(noisy_bayer), dim=0)
      variance    = torch.unsqueeze(totensor_(variance), dim=0)

      # DENOISING THE BAYER IMAGES HERE !
      model.eval()
      raw_image_in = Variable(torch.FloatTensor(noisy_bayer)).cuda()
      raw_image_var= Variable(torch.FloatTensor(variance)).cuda()
      with torch.no_grad():
        raw_image_out = model(raw_image_in, raw_image_var)
      noisy_bayer   = raw_image_in.detach().cpu()
      denoised_bayer= raw_image_out.detach().cpu()
      # DENOISING THE BAYER IMAGES HERE !

      # Flips noisy and denoised bayer images back to original Bayer color pattern.
      if (bayer_pattern == [[1, 2], [2, 3]]):
        pass
      elif (bayer_pattern == [[2, 1], [3, 2]]):
        noisy_bayer = torch.flip(noisy_bayer, dims=[3])
        denoised_bayer = torch.flip(denoised_bayer, dims=[3])
      elif (bayer_pattern == [[2, 3], [1, 2]]):
        noisy_bayer = torch.flip(noisy_bayer, dims=[2])
        denoised_bayer = torch.flip(denoised_bayer, dims=[2])

      # Post-Processing for saving the results
      ccm       = torch.unsqueeze(cam2rgb, dim=0)
      red_g     = torch.unsqueeze(red_gain, dim=0)
      blue_g    = torch.unsqueeze(blue_gain, dim=0)
      # print(noisy_bayer.size())
      noisy_RGB     = process.process(noisy_bayer, red_g, blue_g, ccm)
      denoised_RGB  = process.process(denoised_bayer, red_g, blue_g, ccm)

      out_save = torch.cat((noisy_RGB, denoised_RGB), 3)
      utils.save_image(out_save, args.results_dir + '%04d_%02d.png' % (i + 1, k + 1))
