# ARC_Omniglot
The Attentive Recurrent Comparator (ARC) catches the difference between two characters in the same way that humans do: by iteratively glancing between the images. 

Our experiment follows [ARC implementation of P. Shyam, et. al.](https://arxiv.org/abs/1703.00767) as 20-way one shot classification on the Omniglot dataset.

The authors report 97.75% accuracy for Within Alphabet classification, we achieve 75% (+/- 3%) accuracy in a similar set-up, for several hundred one-shot classifications over the test set. For an alphabet in the wild, we can expect ARC to complete 20-way one-shot classification with 76.1% (+/- 6.4%) accuracy. We conclude that the implemented model has grasped the ability to discriminate between simple visual concepts fairly well.

![Omniglot PCA](https://github.com/ostwind/ARC_Omniglot/blob/master/papers/omniglot_pca.png)

The motivation for ARC's complexity, why Omniglot is more complex than MNIST, and how iterative attention glances 'hone in' on character details, are part of the exploration in fellowship_submission.ipynb.   




## Installation

0. update pytorch to 0.4.1. Do this if you have conda:
    - conda config --add channels soumith
    - conda update pytorch torchvision
1. download the current repo 
2. [download omniglot.npy and omniglot_strokes.npz from here and place it under ./data/](https://drive.google.com/drive/folders/1uGDPpuOy-PXm-Mif3mrnUewGJthkP79A?usp=sharing)
    - omniglot.npy is equivalent to vertically stacking the background and evaluation datasets from Brendan Lake's Omniglot repo
3. run:
   python ARC/train.py --name model
   
   to continue training and read classification accuracies on test set.
