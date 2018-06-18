''' Augment dataset: translation, rotation, flip and shearing 
'''
import os 
from skimage import transform #.transform import rotate
from skimage.io import imsave, imread
import glob
import shutil

def paths(cur_dir):
    return [os.path.join(cur_dir, o) for o in os.listdir(cur_dir) 
                            if os.path.isdir(os.path.join(cur_dir,o))]

def _resize_and_save(path, img, new_height = 32, new_width = 32  ):
    img = transform.resize(img, (new_height, new_width))
    imsave( path, img )

def rotate(letter_path, write_dir_path, alphabet):
    #cur_letter_index = letter_index
    img_paths = []
    for angle in [0]:#, 90, 180, 270]:
        #write_dir, cur_letter_index =  _mkdir(cur_letter_index, alphabet)    
        for sample in glob.glob(letter_path+'/*'):
            img_extension = sample.split('/')[-1]
            img_paths.append(  write_dir_path + img_extension )

            new_img = transform.rotate(imread(sample), angle, resize= False )
            _resize_and_save(   write_dir_path + img_extension, new_img )
            
    return  img_paths

def shear(letter_path, letter_index, alphabet):
    cur_letter_index = letter_index
    for shear_radian in [0.2,0.4,0.6]:
        write_dir, cur_letter_index =  _mkdir(cur_letter_index, alphabet)    
        for sample in glob.glob(letter_path+'/*'):
            
            afine_tf = transform.AffineTransform(shear=shear_radian)
            new_img = transform.warp(imread(sample), inverse_map=afine_tf, cval = 1, mode = 'constant')
            #imsave( write_dir + sample.split('/')[-1], new_img )
            _resize_and_save(  new_img, write_dir + sample.split('/')[-1] )

    return  cur_letter_index    

def flip(letter_path, letter_index, alphabet):
    cur_letter_index = letter_index
    for axis in [1,2]:
        write_dir, cur_letter_index =  _mkdir(cur_letter_index, alphabet)    
        for sample in glob.glob(letter_path+'/*'):
            if axis == 1:
                new_img = imread(sample)[:, ::-1]  
            else:
                new_img = imread(sample)[::-1, :]
            _resize_and_save(  new_img, write_dir + sample.split('/')[-1] )
    return  cur_letter_index    

def identity(letter_path, letter_index, alphabet):
    write_dir, letter_index =  _mkdir(letter_index, alphabet)    
    for sample in glob.glob(letter_path + '/*'):
        _resize_and_save(  imread(sample), write_dir + sample.split('/')[-1] )

    return letter_index

train_directory = './data/train/'
shutil.rmtree(train_directory)
shutil.copytree('./data/train_exemplar', train_directory)

Alphabets = paths(train_directory)
all_letter_paths = []

for alph_ind, alphabet in enumerate(Alphabets):
    alphabet_letter_paths = paths(alphabet)
    #letter_index = len(letters) + 1
    for letter_ind, letter_path in enumerate(alphabet_letter_paths):
        

        # ./data/train/[alphabet name]~[in-alphabet character index]
        write_dir_path =  letter_path.split(
            'character')[0][:-1] + '~' + letter_path.split('character')[-1] + '/'
        os.mkdir(write_dir_path)
        #if letter_path.split('/')[-2] in all_letter_paths
        #all_letter_paths[letter_path.split('/')[-2]].append()

        letter_paths = rotate(letter_path, write_dir_path, alphabet)    
        all_letter_paths += letter_paths

        #letter_index = flip(letter_path,letter_index,alphabet)
        #letter_index = shear(letter_path,letter_index,alphabet)
        
    shutil.rmtree( alphabet )

#import pickle
#pickle.dump( all_letter_paths, open( "manifest.p", "wb" ) ) 
#favorite_color = pickle.load( open( "save.p", "rb" ) )

#print(all_letter_paths)

        



