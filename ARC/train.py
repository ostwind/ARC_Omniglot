import os
import sys 
sys.path.append(os.getcwd())

from model import *
#from util.preprocess import paths 
from ARC import * 

def paths(cur_dir):
    return [os.path.join(cur_dir, o) for o in os.listdir(cur_dir) 
                            if os.path.isdir(os.path.join(cur_dir,o))]


Alphabets = paths('./data/train')


#AlphabetPath = './data/train/Latin/'



batch_size = 8
SupportSet = Alphabet(Alphabets)
SupportLoader = DataLoader(SupportSet, batch_size= batch_size, shuffle=True, num_workers=0)

TestSet = 0
#TestSet = Alphabet(Alphabets)   #, constrain_test_symbol= True)
#TestLoader = DataLoader(TestSet, batch_size=2, shuffle=True, num_workers=1)
#print(len(TestSet) ) 

model = ARC(batch_size = batch_size)
metric = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
stats = dict()
stats['loss'] = []

def train(SupportLoader, TestSet):
    for i in range(3):
        for ind, (samples1, samples2) in  enumerate(SupportLoader):
            test, support, label = GetXY(samples1 , samples2)
            pred = model(support, test)
            
            loss = metric(pred, label)
            stats['loss'].append(loss)
            if ind % 100 == 0:
                print( '%.4f        %s ' %( np.sum(stats['loss'])/100,
                torch.sum(label).item() ) )
                stats['loss'] = []

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #break

train(SupportLoader, TestSet)
