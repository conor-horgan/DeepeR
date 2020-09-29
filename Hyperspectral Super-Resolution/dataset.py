import os
import sys
import random
import numpy as np
import scipy.io
import scipy.signal
from skimage.transform import resize
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class RamanImageDataset(Dataset):
    def __init__(self, image_ids, path, batch_size=2, hr_image_size=64, lr_image_size=16, spectrum_len=500,
                spectrum_shift = 0., spectrum_flip = False, horizontal_flip = False, vertical_flip = False, 
                 rotate = False, patch = False, mixup = False):
        self.image_ids = image_ids
        self.path = path
        self.batch_size = batch_size
        self.hr_image_size = hr_image_size
        self.lr_image_size = lr_image_size
        self.spectrum_len = spectrum_len
        self.spectrum_shift = spectrum_shift
        self.spectrum_flip = spectrum_flip
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate = rotate
        self.patch = patch
        self.mixup = mixup
        self.on_epoch_end()
        
    def load_image(self, id_name):
        input_path =self.path + id_name + ".mat"
        
        output_data = scipy.io.loadmat(input_path)
        output_values = list(output_data.values())
        output_image = output_values[3]
        return output_image
    
    def pad_image(self, image, size, patch):
        if image.shape[0] == size and image.shape[1] == size:
            padded_image = image
        elif image.shape[0] > size and image.shape[1] > size:
            if patch:
                padded_image = self.get_image_patch(image, size)
            else:
                padded_image = self.center_crop_image(image, size)                    
        else:
            padded_image = image
            if padded_image.shape[0] > size:
                if patch:
                    padded_image = self.get_image_patch(padded_image, size)
                else:
                    padded_image = self.center_crop_image(padded_image, size) 
            else:           
                pad_before = int(np.floor((size - padded_image.shape[0])/2))
                pad_after = int(np.ceil((size - padded_image.shape[0])/2))
                padded_image = np.pad(padded_image, ((pad_before, pad_after), (0,0), (0, 0)), 'reflect')

            if padded_image.shape[1] > size:
                if patch:
                    padded_image = self.get_image_patch(padded_image, size)
                else:
                    padded_image = self.center_crop_image(padded_image, size) 
            else:           
                pad_before = int(np.floor((size - padded_image.shape[1])/2))
                pad_after = int(np.ceil((size - padded_image.shape[1])/2))
                padded_image = np.pad(padded_image, ((0,0), (pad_before, pad_after), (0, 0)), 'reflect')

        return padded_image

    def get_image_patch(self, image, patch_size):                   
        if image.shape[0] > patch_size:
            start_idx_x = int(np.round(np.random.random() * (image.shape[0]-patch_size)))
            end_idx_x = start_idx_x + patch_size
        else:
            start_idx_x = 0
            end_idx_x = image.shape[0]

        if image.shape[1] > patch_size:
            start_idx_y = int(np.round(np.random.random() * (image.shape[1]-patch_size)))
            end_idx_y = start_idx_y + patch_size
        else:
            start_idx_y = 0
            end_idx_y = image.shape[1]

        image_patch = image[start_idx_x:end_idx_x,start_idx_y:end_idx_y,:]
        return image_patch

    def center_crop_image(self, image, image_size):
        cropped_image = image
        if image.shape[0] > image_size:
            dif = int(np.floor((image.shape[0] - image_size)/2))
            cropped_image = cropped_image[dif:image_size+dif,:,:]

        if image.shape[1] > image_size:
            dif = int(np.floor((image.shape[1] - image_size)/2))
            cropped_image = cropped_image[:,dif:image_size+dif,:]
        return cropped_image
     
    def flip_axis(self, image, axis):
        if np.random.random() < 0.5:
            image = np.asarray(image).swapaxes(axis, 0)
            image = image[::-1, ...]
            image = image.swapaxes(0, axis)
        return image
        
    def rotate_spectral_image(self, image):
        rotation_extent = np.random.random()
        if rotation_extent < 0.25:
            rotation = 1
        elif rotation_extent < 0.5:
            rotation = 2
        elif rotation_extent < 0.75:
            rotation = 3
        else:
            rotation = 0
        image = np.rot90(image, rotation)
        return image
    
    def shift_spectrum(self, image, shift_range):
        shifted_spectrum_image = image
        spectrum_shift_range = int(np.round(shift_range*image.shape[2]))
        if spectrum_shift_range > 0:
            shifted_spectrum_image = np.pad(image[:,:,spectrum_shift_range:], ((0,0), (0,0), (0,abs(spectrum_shift_range))), 'reflect')
        elif spectrum_shift_range < 0:
            shifted_spectrum_image = np.pad(image[:,:,:spectrum_shift_range], ((0,0), (0,0), (abs(spectrum_shift_range), 0)), 'reflect')
        return shifted_spectrum_image
    
    def spectrum_padding(self, image, spectrum_length):
        if image.shape[-1] == spectrum_length:
            padded_spectrum_image = image
        elif image.shape[-1] > spectrum_length:
            padded_spectrum_image = image[:,:,0:spectrum_length]
        else:
            padded_spectrum_image = np.pad(image, ((0,0), (0,0), (0, spectrum_length - image.shape[-1])), 'reflect')
        return padded_spectrum_image
    
    def image_mixup(self, image1, image2, alpha):
        lam = np.random.beta(alpha, alpha)
        image = (lam * image1) + ((1 - lam) * image2)
        return image
    
    def normalise_image(self, image):
        image_max = np.tile(np.amax(image),image.shape)
        normalised_image = np.divide(image,image_max)
        return normalised_image
    
    def downsample_image(self, image, scale = 4):
        if scale >= 4:
            start_idx = np.random.randint(1,scale-1)
        else:
            start_idx = 1 
        downsampled_image = image[start_idx::scale,start_idx::scale,:]
        return downsampled_image  
    
    def __getitem__(self, idx):
        image_size_ratio = self.hr_image_size // self.lr_image_size

        outputimg = self.load_image(self.image_ids[idx])
        
        mixup_on = False
        if self.mixup:
            if np.random.random() < 0.5:
                image_idx = int(np.round(np.random.random() * (len(self.image_ids)-1)))
                image2 = self.load_image(self.image_ids[image_idx])
                mixup_on = True

        # --------------- Image Data Augmentations --------------- 
        outputimg = self.pad_image(outputimg, self.hr_image_size, self.patch)
        if mixup_on:
            image2 = self.pad_image(image2, self.hr_image_size, self.patch)

        if self.horizontal_flip:    
            outputimg = self.flip_axis(outputimg, 1)
            if mixup_on:
                image2 = self.flip_axis(image2, 1)

        if self.vertical_flip:    
            outputimg = self.flip_axis(outputimg, 0)
            if mixup_on:
                image2 = self.flip_axis(image2, 0)

        if self.rotate:
            outputimg = self.rotate_spectral_image(outputimg)
            if mixup_on:
                image2 = self.rotate_spectral_image(image2)

        # --------------- Spectral Data Augmentations --------------- 
        if self.spectrum_shift != 0.0:
            shift_range = np.random.uniform(-self.spectrum_shift, self.spectrum_shift)
            outputimg = self.shift_spectrum(outputimg, shift_range)
            if mixup_on:
                image2 = self.shift_spectrum(image2, shift_range)

        outputimg = self.spectrum_padding(outputimg, self.spectrum_len)
        if mixup_on:
            image2 = self.spectrum_padding(image2, self.spectrum_len)

        if self.spectrum_flip:    
            if np.random.random() < 0.5:
                outputimg = self.flip_axis(outputimg, 2)
                if mixup_on:
                    image2 = self.flip_axis(image2, 2)
        
        # --------------- Mixup --------------- 
        if mixup_on:
            outputimg = self.image_mixup(outputimg, image2, 0.2)

        # --------------- Normalisation and Downsampling --------------- 
        outputimg = self.normalise_image(outputimg)
        inputimg = self.downsample_image(outputimg, image_size_ratio)
        
        outputimg = np.moveaxis(outputimg, -1, 0)
        inputimg = np.moveaxis(inputimg, -1, 0)

        sample = {'input_image': inputimg, 'output_image': outputimg}
        
        return sample
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return len(self.image_ids)
