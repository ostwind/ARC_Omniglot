# ARC_Omniglot
Challenge submission for Fellowship.AI 

## Please see fellowship_submission.ipynb for analysis

## To run the model

0. update pytorch to 0.4.1. Do this if you have conda:
    - conda config --add channels soumith
    - conda update pytorch torchvision
1. download the current repo 
2. [download omniglot.npy and omniglot_strokes.npz from here and place it under ./data/](https://drive.google.com/drive/folders/1uGDPpuOy-PXm-Mif3mrnUewGJthkP79A?usp=sharing)
    - omniglot.npy is equivalent to vertically stacking the background and evaluation datasets from Brendan Lake's Omniglot repo
3. run:
   python ARC/train.py --name model
   
   to continue training and read classification accuracies on test set.
