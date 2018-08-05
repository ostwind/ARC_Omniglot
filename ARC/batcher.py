"""
taken and modified from https://github.com/pranv/ARC
"""
import os
import numpy as np
from numpy.random import choice
import torch
from torch.autograd import Variable

from scipy.misc import imresize as resize
from image_augmenter import ImageAugmenter

class Omniglot(object):
    def __init__(self, path=os.path.join('data', 'omniglot.npy'), batch_size=60):
        """
        batch_size: the output is (2 * batch size, 1, image_size, image_size)
                    X[i] & X[i + batch_size] are the pair
        image_size: size of the image
        data_split: in number of alphabets, current split is 30 is for training, 20 for testing
        within_alphabet: for verfication task, when 2 characters are sampled to form a pair,
                        this flag specifies if should they be from the same alphabet/language
        ---------------------
        Data Augmentation Parameters:
            flip: here flipping both the images in a pair
            scale: x would scale image by + or - x%
            rotation_deg
            shear_deg
            translation_px: in both x and y directions
        """
        image_size = 32
        chars = np.load(path)

        # resize the images
        resized_chars = np.zeros((1623, 20, image_size, image_size), dtype='uint8')
        for i in range(1623):
            for j in range(20):
                resized_chars[i, j] = resize(chars[i, j], (image_size, image_size))
        chars = resized_chars

        self.mean_pixel = chars.mean() / 255.0  # used later for mean subtraction

        # starting index of each alphabet in a list of chars
        a_start = [0, 20, 49, 75, 116, 156, 180, 226, 240, 266, 300, 333, 355, 381,
                   424, 448, 496, 518, 534, 586, 633, 673, 699, 739, 780, 813,
                   827, 869, 892, 909, 964, 984, 1010, 1036, 1062, 1088, 1114,
                   1159, 1204, 1245, 1271, 1318, 1358, 1388, 1433, 1479, 1507,
                   1530, 1555, 1597]

        # size of each alphabet (num of chars)
        a_size = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22,
                  16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26,
                  26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

        # each alphabet/language has different number of characters.
        # in order to uniformly sample all characters, we need weigh the probability
        # of sampling a alphabet by its size. p is that probability
        def size2p(size):
            s = np.array(size).astype('float64')
            return s / s.sum()

        self.size2p = size2p

        self.data = chars
        self.a_start = a_start
        self.a_size = a_size
        self.image_size = image_size
        self.batch_size = batch_size

        flip = True
        scale = 0.2
        rotation_deg = 20
        shear_deg = 10
        translation_px = 5
        self.augmentor = ImageAugmenter(image_size, image_size,
                                        hflip=flip, vflip=flip,
                                        scale_to_percent=1.0 + scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                                        translation_x_px=translation_px, translation_y_px=translation_px)

    def fetch_batch(self, part):
        """
            This outputs batch_size number of pairs
            Thus the actual number of images outputted is 2 * batch_size
            Say A & B form the half of a pair
            The Batch is divided into 4 parts:
                Dissimilar A 		Dissimilar B
                Similar A 			Similar B

            Corresponding images in Similar A and Similar B form the similar pair
            similarly, Dissimilar A and Dissimilar B form the dissimilar pair

            When flattened, the batch has 4 parts with indices:
                Dissimilar A 		0 - batch_size / 2
                Similar A    		batch_size / 2  - batch_size
                Dissimilar B 		batch_size  - 3 * batch_size / 2
                Similar B 			3 * batch_size / 2 - batch_size
        """
        pass

class Batcher(Omniglot):
    def __init__(self, path=os.path.join('data', 'omniglot.npy'), 
    cuda = False, batch_size=60):
        Omniglot.__init__(self, path, batch_size)

        a_start = self.a_start
        a_size = self.a_size

        # slicing indices for splitting a_start & a_size
        split_ind = 30 
        starts, sizes = {}, {}
        starts['train'], starts['test'] = a_start[:split_ind],  a_start[split_ind:]
        sizes['train'], sizes['test'] = a_size[:split_ind], a_size[split_ind:]

        size2p = self.size2p

        p = {}
        p['train'], p['test'] = size2p(sizes['train']), size2p(sizes['test'])

        self.image_size = 32
        self.starts = starts
        self.sizes = sizes
        self.p = p
        self.use_cuda = cuda

    def fetch_batch(self, part = None, batch_size: int = None):
        if batch_size is None:
           batch_size = self.batch_size
        
        if part == 'train':
            X, Y = self._fetch_batch(part, batch_size)
        else: 
            X, Y = self._fetch_eval(alphabet_idx = part)

        X = X - self.mean_pixel
        X = X[:, np.newaxis]
        X = X.astype("float32")
        X = torch.from_numpy(X).view(2*batch_size, self.image_size, self.image_size)

        X1 = X[:batch_size]  # (B, h, w)
        X2 = X[batch_size:]  # (B, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, h, w)
        Y = torch.from_numpy(Y)

        if self.use_cuda:
            X, Y = X.cuda(), Y.cuda()
        return X, Y

    def _fetch_eval(self, alphabet_idx):
        ''' 
            To load a batch of test data into the model so that 20-way one-shot classification 
            can be conducted, match each test image with every image in support set:
            
            Test    Support Set      Labels
            Char 1 | Char 1          1
            Char 1 | Char 2          0
            ...
            Char 1 | Char 20         0
            --------------
            Char 2 | Char 1          0
            Char 2 | Char 2          1
            ...
            Char 2 | Char 20         0
            --------------
            Char 3 | Char 1          0
            ...
            Char 3 | Char 20         0
            
            The test and support sets are outputted from  _fetch_eval() 
            in a single column, then matched horizontally in fetch_batch() (like above  ) 
        '''

        batch_size = 120 
        starts, sizes = self.starts['test'], self.sizes['test']

        # select alphabet to do 20-way 1 shot evaluation
        if not alphabet_idx:
            alphabet_idx = choice( len(starts[:-1]) )              
        
        alphabet_range = range( 
            starts[alphabet_idx], starts[alphabet_idx] + sizes[alphabet_idx] )

        # select 20 characters for support set, select 3 for test set from this support 
        support_idx = choice( alphabet_range, size = 20, replace = False )
        test_idx = support_idx[:3] # choose 3 test characters for this batch
        test_idx =  np.sort( np.repeat( test_idx, 20 ) )
        support_idx = np.tile( support_idx, 3 )
        
        y = np.array(test_idx == support_idx).astype('int32')
        X = np.zeros((batch_size, self.image_size, self.image_size), dtype='uint8')

        X[:batch_size//2] = self.data[ test_idx, choice(20, 1, replace=False)]
        X[batch_size // 2:] = self.data[support_idx, choice(20, 1, replace=False)]
        
        X = X / 255.0
        return X, y

    def _fetch_batch(self, part, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        starts, sizes = self.starts[part], self.sizes[part] 
        p = self.p[part]
        num_alphabets = len(starts)

        X = np.zeros((2 * batch_size, self.image_size, self.image_size), dtype='uint8')
        for i in range(batch_size // 2):
            # choose similar chars
            same_idx = choice(range(starts[0], starts[-1] + sizes[-1]))

            # choose dissimilar chars within alphabet
            alphabet_idx = choice(num_alphabets, p=p)
            char_offset = choice(sizes[alphabet_idx], 2, replace=False)
            diff_idx = starts[alphabet_idx] + char_offset
            
            X[i], X[i + batch_size] = self.data[diff_idx, choice(20, 2)]
            X[i + batch_size // 2], X[i + 3 * batch_size // 2] = self.data[
                same_idx, choice(20, 2, replace=False)]

        y = np.zeros((batch_size, 1), dtype='int32')
        y[:batch_size // 2] = 0
        y[batch_size // 2:] = 1

        if part == 'train':
          X = self.augmentor.augment_batch(X)
        else:
          X = X / 255.0

        return X, y

