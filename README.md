# Unprocessing Images for Learned Raw Denoising, CVPR'19 (Unofficial PyTorch Code)
Unofficial PyTorch implementation of the paper - Unprocessing Images for Learned Raw Denoising, CVPR'19, Tim Brooks, Ben Mildenhall, Tianfan Xue, Jiawen Chen, Dillon Sharlet, Jonathan T. Barron. 

This implementation is heavily borrowed from the offical Tensorflow code, which can be picked from [here](https://github.com/google-research/google-research/tree/master/unprocessing). 

Please ensure that you cite the paper if you use this code:
```
@inproceedings{brooks2019unprocessing,
  title={Unprocessing Images for Learned Raw Denoising},
  author={Brooks, Tim and Mildenhall, Ben and Xue, Tianfan and Chen, Jiawen and Sharlet, Dillon and Barron, Jonathan T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
}
```
### Requirements
The code is tested on Python 3.7, PyTorch 1.1.0, TorchVision 0.3.0, but lower versions are also likely to work. During training on a single NVidia GTX1080 GPU, keeping a batch-size of 16 and images cropped to 256x256, the memory consumption was found to be under 4Gb. 

### Training
In the paper, the authors use the [MIRFlickr](https://press.liacs.nl/mirflickr/) dataset for training. You can use the entire set of 1M images (as done in the paper) or simply take the smaller subset of 25k images which I did for my training process. I then divided the set into training and validation sets containing 23750 and 1250 images respectively. I then manually filtered out images with less than 256x256 resolution from the training set, effectively giving 23442 images for training. 



