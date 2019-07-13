#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim

#Tools lib
import numpy as np
from time import sleep
import os
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import utils as vutils
from dataloader import load_data as DA
from dataloader import process
#Models lib
from models import *
#Metrics lib
from metrics import calc_psnr, calc_ssim

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
    A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True, default=None,
        help="Location at which to save the model, logs and checkpoints.")
    parser.add_argument("--load_model", type=str, required=False, default=None,
        help="Location from which any pre-trained model needs to be loaded.")
    parser.add_argument("--train_dir", type=str, required=True, default=None,
        help="Directory containing source JPG images for training.")
    parser.add_argument("--test_dir", type=str, required=True, default=None,
        help="Directory containing source JPG images for testing/validating.")
    parser.add_argument("--load_size", type=str, default=None,
        help="Width and height to resize training and testing frames. Must be a multiple of 16. None for no resizing")
    parser.add_argument("--crop_size", type=str, default="[256, 256]",
        help="Width and height to crop training and testing frames. Must be a multiple of 16")
    parser.add_argument("--batch_size", type=int, default=16,
        help="Batch size to train the model.")
    parser.add_argument("--epochs", type=int, default=400,
        help="No of epochs to train and validate the model.")
    parser.add_argument("--epoch_start", type=int, default=0,
        help="Epoch to start training the model from.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
        help="Learning rate for the model.")
    parser.add_argument('--skip_validation', action='store_true',
    	help='Whether to skip validation in the training process?')
    parser.add_argument('--gpu_id', type=int, default=3,
        help='Select the gpu_id to run the code on')

    return parser.parse_args()

def validate(model, inputs, labels):

    model.eval()

    raw_image_in   = Variable(torch.FloatTensor(inputs['noisy_img'])).cuda()
    raw_image_var  = Variable(torch.FloatTensor(inputs['variance'])).cuda()
    raw_image_gt   = Variable(torch.FloatTensor(labels)).cuda()
    red_gain       = Variable(torch.FloatTensor(inputs['red_gain'])).cuda()
    blue_gain      = Variable(torch.FloatTensor(inputs['blue_gain'])).cuda()
    cam2rgb        = Variable(torch.FloatTensor(inputs['cam2rgb'])).cuda()

    with torch.no_grad():
        raw_image_out = model(raw_image_in, raw_image_var)
    
    # Process RAW images to RGB
    rgb_image_in  = process.process(raw_image_in, red_gain, blue_gain, cam2rgb)
    rgb_image_out = process.process(raw_image_out, red_gain, blue_gain, cam2rgb)
    rgb_image_gt  = process.process(raw_image_gt, red_gain, blue_gain, cam2rgb)

    rgb_image_out = rgb_image_out[0, :, :, :].cpu().data.numpy().transpose((1, 2, 0))
    rgb_image_out = np.array(rgb_image_out*255.0, dtype = 'uint8')
    rgb_image_gt  = rgb_image_gt[0, :, :, :].cpu().data.numpy().transpose((1, 2, 0))
    rgb_image_gt  = np.array(rgb_image_gt*255.0, dtype = 'uint8')
    # print(np.shape(rgb_image_out), np.shape(rgb_image_gt))

    cur_psnr = calc_psnr(rgb_image_out, rgb_image_gt)
    cur_ssim = calc_ssim(rgb_image_out, rgb_image_gt)
    
    return cur_psnr, cur_ssim

def train(model, optimizer, inputs, labels):

    model.train()

    raw_image_in   = Variable(torch.FloatTensor(inputs['noisy_img'])).cuda()
    raw_image_var  = Variable(torch.FloatTensor(inputs['variance'])).cuda()
    raw_image_gt   = Variable(torch.FloatTensor(labels)).cuda()
    red_gain       = Variable(torch.FloatTensor(inputs['red_gain'])).cuda()
    blue_gain      = Variable(torch.FloatTensor(inputs['blue_gain'])).cuda()
    cam2rgb        = Variable(torch.FloatTensor(inputs['cam2rgb'])).cuda()

    optimizer.zero_grad()
    raw_image_out = model(raw_image_in, raw_image_var)

    # Process RAW images to RGB
    rgb_image_in  = process.process(raw_image_in, red_gain, blue_gain, cam2rgb)
    rgb_image_out = process.process(raw_image_out, red_gain, blue_gain, cam2rgb)
    rgb_image_gt  = process.process(raw_image_gt, red_gain, blue_gain, cam2rgb)

    loss  = F.l1_loss(rgb_image_out, rgb_image_gt.detach()) 
    loss.backward()
    optimizer.step()

    return loss.item(), rgb_image_in, rgb_image_out, rgb_image_gt


if __name__ == '__main__':
    # Load args
    args  = get_arguments()
    torch.cuda.set_device(args.gpu_id)
    args.logs_dir    = args.model_dir + '/logs/'
    args.visuals_dir = args.model_dir + '/visuals/'
    args.nets_dir    = args.model_dir + '/nets/'
    if not os.path.exists( args.logs_dir):
    	os.makedirs(args.logs_dir)
    if not os.path.exists( args.visuals_dir):
    	os.makedirs(args.visuals_dir)
    if not os.path.exists( args.nets_dir):
    	os.makedirs(args.nets_dir)
    # Initialize log writer
    logger= SummaryWriter(args.logs_dir)
    # Create generator model
    model = Generator().cuda()
    model = nn.DataParallel(model, device_ids=[args.gpu_id]).cuda()
    if args.load_model is not None:
        print('Loading pre-trained checkpoint %s'% args.load_model)
        model_psnr = torch.load(args.load_model)['avg_psnr']
        model_ssim = torch.load(args.load_model)['avg_ssim']
        print('Avg. PSNR and SSIM values recorded from the checkpoint: %f, %f' % (model_psnr, model_ssim))
        model_state_dict = torch.load(args.load_model)['state_dict']
        model.load_state_dict(model_state_dict)

    # Create optimizer
    optimizer  = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    # Get training and validation data (if validation needed)
    tr_input_list = sorted([file for file in os.listdir(args.train_dir) if file.endswith('.jpg')])
    val_input_list= sorted([file for file in os.listdir(args.test_dir) if file.endswith('.jpg')])

	# Create train loader
    TrainImgLoader = torch.utils.data.DataLoader(DA.loadImgs(args, tr_input_list, mode='train'),
                                                 batch_size  = args.batch_size, 
                                                 shuffle     = True, 
                                                 num_workers = 8, 
                                                 drop_last   = False)
    # Create val loader
    ValImgLoader   = torch.utils.data.DataLoader(DA.loadImgs(args, val_input_list, mode='val'),
                                                 batch_size  = 1,
                                                 shuffle     = False, 
                                                 num_workers = 8, 
                                                 drop_last   = False)
    
    # Validate the network at epoch 0 (if needed)
    avg_psnr    = 0.0
    avg_ssim    = 0.0
    avg_tr_loss = 0.0
    if not args.skip_validation:
        epoch           = args.epoch_start
        cumulative_psnr = 0
        cumulative_ssim = 0
        count_idx       = 0
        tbar = tqdm(ValImgLoader)
        for batch_idx, (inputs, labels) in enumerate(tbar):
            count_idx = count_idx + 1
            cur_psnr, cur_ssim = validate(model, inputs, labels)
            # print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
            avg_psnr = cumulative_psnr / count_idx
            avg_ssim = cumulative_ssim / count_idx
            desc = 'Validation: Epoch %d, Avg. PSNR = %.4f and SSIM = %.4f' % (epoch, avg_psnr, avg_ssim)
            tbar.set_description(desc)
            tbar.update()
            # sleep(0.01)
        logger.add_scalar('Validation/avg_psnr', avg_psnr, epoch)
        logger.add_scalar('Validation/avg_ssim', avg_ssim, epoch)


    # Train the network for the given number of epochs, and
    # Validate it every epoch as well
    glb_iter  = 0
    for epoch in range(args.epoch_start+1, args.epoch_start+args.epochs+1):
        tr_loss         = 0
        cumulative_psnr = 0
        cumulative_ssim = 0
        count_idx       = 0
        # adjust_learning_rate(args, optimizer, epoch)
        # Train the network for the given epoch
        tbar = tqdm(TrainImgLoader)
        for batch_idx, (inputs, labels) in enumerate(tbar):
            count_idx = count_idx + 1
            loss, rgb_image_in, rgb_image_out, rgb_image_gt  = train(model, optimizer, inputs, labels)
            tr_loss  = tr_loss + loss
            logger.add_scalar('Train/loss', loss, glb_iter)
            if glb_iter%400 == 0:
                in_save     = rgb_image_in.detach().cpu()
                out_save    = rgb_image_out.detach().cpu()
                gt_save     = rgb_image_gt.detach().cpu()
                res_save    = torch.cat((in_save, out_save, gt_save), 2)
                vutils.save_image(res_save, args.visuals_dir + '/visual' + str(epoch) + '_' + str(glb_iter) + '.jpg')
            glb_iter = glb_iter+1;
            avg_tr_loss = tr_loss / count_idx
            desc = 'Training  : Epoch %d, Avg. Loss = %.5f' %(epoch, avg_tr_loss)
            tbar.set_description(desc)
            tbar.update()
            # sleep(0.01)
        logger.add_scalar('Train/avg_loss', avg_tr_loss, epoch)


        # Validate the network for the given epoch
        if not args.skip_validation:
            count_idx = 0
            tbar = tqdm(ValImgLoader)
            for batch_idx, (inputs, labels) in enumerate(tbar):
                count_idx = count_idx + 1
                cur_psnr, cur_ssim = validate(model, inputs, labels)
                # print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
                cumulative_psnr += cur_psnr
                cumulative_ssim += cur_ssim
                avg_psnr = cumulative_psnr / count_idx
                avg_ssim = cumulative_ssim / count_idx
                desc = 'Validation: Epoch %d, Avg. PSNR = %.4f and SSIM = %.4f' % (epoch, avg_psnr, avg_ssim)
                tbar.set_description(desc)
                tbar.update()
                # sleep(0.01)
            logger.add_scalar('Validation/avg_psnr', avg_psnr, epoch)
            logger.add_scalar('Validation/avg_ssim', avg_ssim, epoch)


        # Save the network per epoch with performance metrics as well
        savefilename = args.nets_dir+'/checkpoint'+'_'+str(epoch)+'.tar'
        torch.save({
             'epoch': epoch,
             'avg_psnr': avg_psnr,
             'avg_ssim': avg_ssim, 
             'state_dict': model.state_dict(),
             'avg_tr_loss': avg_tr_loss}, savefilename)

