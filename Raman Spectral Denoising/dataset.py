import os
import sys
import random
import numpy as np
import scipy.io
import scipy.signal
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class RamanDataset(Dataset):
    def __init__(self, inputs, outputs, batch_size=64,spectrum_len=500, spectrum_shift=0., 
                 spectrum_window=False, horizontal_flip=False, mixup=False):
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.spectrum_len = spectrum_len
        self.spectrum_shift = spectrum_shift
        self.spectrum_window = spectrum_window
        self.horizontal_flip = horizontal_flip
        self.mixup = mixup
        self.on_epoch_end()     
        
    def pad_spectrum(self, input_spectrum, spectrum_length):
        if len(input_spectrum) == spectrum_length:
            padded_spectrum = input_spectrum
        elif len(input_spectrum) > spectrum_length:
            padded_spectrum = input_spectrum[0:spectrum_length]
        else:
            padded_spectrum = np.pad(input_spectrum, ((0,spectrum_length - len(input_spectrum)),(0,0)), 'reflect')

        return padded_spectrum
    
    def window_spectrum(self, input_spectrum, start_idx, window_length):
        if len(input_spectrum) <= window_length:
            output_spectrum = input_spectrum
        else:
            end_idx = start_idx + window_length
            output_spectrum = input_spectrum[start_idx:end_idx]

        return output_spectrum
            
    def flip_axis(self, x, axis):
        if np.random.random() < 0.5:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
        return x
    
    def shift_spectrum(self, x, shift_range):
        x = np.expand_dims(x,axis=-1)
        shifted_spectrum = x
        spectrum_shift_range = int(np.round(shift_range*len(x)))
        if spectrum_shift_range > 0:
            shifted_spectrum = np.pad(x[spectrum_shift_range:,:], ((0,abs(spectrum_shift_range)), (0,0)), 'reflect')
        elif spectrum_shift_range < 0:
            shifted_spectrum = np.pad(x[:spectrum_shift_range,:], ((abs(spectrum_shift_range), 0), (0,0)), 'reflect')
        return shifted_spectrum
    
    def mixup_spectrum(self, input_spectrum1, input_spectrum2, output_spectrum1, output_spectrum2, alpha):
        lam = np.random.beta(alpha, alpha)
        input_spectrum = (lam * input_spectrum1) + ((1 - lam) * input_spectrum2)
        output_spectrum = (lam * output_spectrum1) + ((1 - lam) * output_spectrum2)
        return input_spectrum, output_spectrum
            
    def __getitem__(self, index):       
        input_spectrum = self.inputs[index]
        output_spectrum = self.outputs[index]
        
        mixup_on = False
        if self.mixup:
            if np.random.random() < 0.5:
                spectrum_idx = int(np.round(np.random.random() * (len(self.inputs)-1)))
                input_spectrum2 = self.inputs[spectrum_idx]
                output_spectrum2 = self.outputs[spectrum_idx]
                mixup_on = True

        if self.spectrum_window:
            start_idx = int(np.floor(np.random.random() * (len(input_spectrum)-self.spectrum_len)))
            input_spectrum = self.window_spectrum(input_spectrum, start_idx, self.spectrum_len)
            output_spectrum = self.window_spectrum(output_spectrum, start_idx, self.spectrum_len)
            if mixup_on:
                input_spectrum2 = self.window_spectrum(input_spectrum2, start_idx, self.spectrum_len)
                output_spectrum2 = self.window_spectrum(output_spectrum2, start_idx, self.spectrum_len)

        input_spectrum = self.pad_spectrum(input_spectrum, self.spectrum_len)
        output_spectrum = self.pad_spectrum(output_spectrum, self.spectrum_len)
        if mixup_on:
            input_spectrum2 = self.pad_spectrum(input_spectrum2, self.spectrum_len)
            output_spectrum2 = self.pad_spectrum(output_spectrum2, self.spectrum_len)

        if self.spectrum_shift != 0.0:
            shift_range = np.random.uniform(-self.spectrum_shift, self.spectrum_shift)
            input_spectrum = self.shift_spectrum(input_spectrum, shift_range)
            output_spectrum = self.shift_spectrum(output_spectrum, shift_range)
            if mixup_on:
                input_spectrum2 = self.shift_spectrum(input_spectrum2, shift_range)
                output_spectrum2 = self.shift_spectrum(output_spectrum2, shift_range)
        else:
            input_spectrum = np.expand_dims(input_spectrum, axis=-1)
            output_spectrum = np.expand_dims(output_spectrum, axis=-1)
            if mixup_on:
                input_spectrum2 = np.expand_dims(input_spectrum2, axis=-1)
                output_spectrum2 = np.expand_dims(output_spectrum2, axis=-1)

        if self.horizontal_flip:    
            if np.random.random() < 0.5:
                input_spectrum = self.flip_axis(input_spectrum, 0)
                output_spectrum = self.flip_axis(output_spectrum, 0)
                if mixup_on:
                    input_spectrum2 = self.flip_axis(input_spectrum2, 0)
                    output_spectrum2 = self.flip_axis(output_spectrum2, 0)

        if mixup_on:
            input_spectrum, output_spectrum = self.mixup_spectrum(input_spectrum, input_spectrum2, output_spectrum, output_spectrum2, 0.2)
            
        input_spectrum = input_spectrum/np.amax(input_spectrum)
        output_spectrum = output_spectrum/np.amax(output_spectrum)
        
        input_spectrum = np.moveaxis(input_spectrum, -1, 0)
        output_spectrum = np.moveaxis(output_spectrum, -1, 0)
        
        sample = {'input_spectrum': input_spectrum, 'output_spectrum': output_spectrum}
        
        return sample
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return len(self.inputs)
