import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import random

class Alphabet(Dataset):
    """Retrieve all images in alphabet"""
    def __init__(self, alphabet_dirs, constrain_test_symbol = False, transform=None):
        """
        Args:
            alphabet_dir (string): select a language
            constrain_test_symbol (optional): constrain samples from a specific letter
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        self.letter_paths = alphabet_dirs
        # for alphabet_dir in alphabet_dirs:
        #     self.letter_paths += [os.path.join(alphabet_dir, o) for o in os.listdir(alphabet_dir) 
        #                     if os.path.isdir(os.path.join(alphabet_dir,o))]
        
        if constrain_test_symbol:
            self.letter_paths = [ np.random.choice(self.letter_paths) ]

        self.all_samples = [ ]
        for letter in self.letter_paths:
            self.all_samples += glob.glob(letter + '/*' )
        self.all_samples.sort()

    def __len__(self):
        return len(self.all_samples)

    def get_sample(self, idx):
        image = io.imread( self.all_samples[idx] )
        sample = {'image': image.astype(np.uint8),
        'letter': self.all_samples[idx].split('/')[-2] }
        return sample

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        
        if len(self.all_samples) >= idx+2:
            sample2 = self.get_sample(idx + 1)
        
        else:
            sample2 = self.get_sample(idx - 1)
        
        return sample, sample2 

def GetXY(ImgSet1, ImgSet2):
    '''
        Image 1      Image 2

        DISSIMILAR | DISSIMILAR
        -----------|-----------
        SIMILAR    | SIMILAR
    '''
    
    
    # take symbol from same character class as test set, append to support set  
    images1,  letters1 = ImgSet1['image'],  ImgSet1['letter'], 
    
    half_batch_size = images1.shape[0]//2
    
    combined = list(zip( ImgSet2['image'], ImgSet2['letter'] ) )
    random.shuffle(combined)
    mixed_img, mixed_letter = zip(*combined)
    images2, letters2 = ImgSet2['image'], ImgSet2['letter']
    
    images2[:half_batch_size] = torch.stack(mixed_img[:half_batch_size])   
    letters2[:half_batch_size] = mixed_letter[:half_batch_size]  
    
    #build label 
    label = np.array( letters1 ) == np.array( letters2 )
    label = torch.from_numpy(label.astype(int))

    # additional channel for in-channel (req. conv layer)
    images1  = images1.unsqueeze(1).float()#torch.from_numpy( np.array(images1) ).unsqueeze(1).float()     
    images2 = images2.unsqueeze(1).float()

    return images1, images2, label

# def sample_pair( letter_paths, similar = True):
#     #if similar:
#     return 

# def loader(dir = './data/train/'):
#     #mylist = [f for f in glob.glob("%s/*.png" %dir )]
#     letter_paths =  os.listdir(dir)
#     print(x)

#loader()
