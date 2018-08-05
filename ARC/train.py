import os
import argparse
import sys 
sys.path.append(os.getcwd())

from model import *
from ARC import * 
from batcher import Batcher

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=60, help='input batch size')
parser.add_argument('--CauchyKer_size', type=int, default=4, help='the height / width of glimpse seen by ARC')
parser.add_argument('--hidden_size', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--num_glimpses', type=int, default=8, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='name for checkpoints of this model, R/W directed here')
arg = parser.parse_args()

model = ARC(batch_size = arg.batch_size, CauchyKerSize = arg.CauchyKer_size,
hidden_size = arg.hidden_size, num_glimpses = arg.num_glimpses)
print('running on %s devices' %( torch.cuda.device_count() ) )
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = arg.lr)
stats = dict()
stats['loss'] = []

loader = Batcher(batch_size= arg.batch_size)

if arg.name is not None:
    model_name = './'+arg.name + '.pth'
else: 
    model_name = './%s_%s.pth' %(arg.glimpse_size, arg.lr)

if os.path.exists(model_name):
    model.load_state_dict(torch.load( model_name ))

def accuracy(pred, target) -> int:
    hard_pred = (pred > 0.5).int()
    correct = (hard_pred == target).sum().item()
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def one_shot_eval(pred, truth):
    corrects, error_margin = 0, 0
    if   truth[ pred[:20].max(0)[1] ] == 1:
        corrects += 1
    if   truth[ pred[20:40].max(0)[1]+20 ] == 1:
        corrects += 1
    if   truth[ pred[40:].max(0)[1]+40 ] == 1 :
        corrects += 1
    return corrects 

def train(loader):
    model.train()
    for ind in range(1000*1000):
        X, Y = loader.fetch_batch("train")
        if use_cuda:
            X, Y = X.cuda(), Y.cuda()
        pred = model(X)
        loss = criterion(pred, Y.float())
        stats['loss'].append(loss)
        
        if ind % 2000 == 0:
            print( 'training loss: %.3f   similarity train acc: %d    epoch:  %s' %(
            np.sum( stats['loss'] )/len(stats['loss']),
            accuracy(pred, Y), 
            ind ) )
            stats['loss'] = []
            
            with torch.no_grad():
                one_shot_by_alphabet = []
                for alphabet in range(20):
                    eval_acc = 0 
                    for test_ind in range(100):
                        # remove 'part' argument to randomly choose alphabet
                        X_val, Y_val = loader.fetch_batch()#part = alphabet)
                        if use_cuda:
                            X_val, Y_val = X_val.cuda(), Y_val.cuda()
            
                        pred_val = model(X_val, test_ind)
                        eval_acc += one_shot_eval(pred_val, Y_val )
                    
                    one_shot_by_alphabet.append(  round(eval_acc/300,2) )
                    print('Alphabet %s accuracy: %.3f' %(alphabet+30, eval_acc/300))
            
            print(one_shot_by_alphabet)
            print('\n')
            #exit()

            if ind % 2000 == 0:
                torch.save(model.state_dict(), model_name )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
train(loader)
