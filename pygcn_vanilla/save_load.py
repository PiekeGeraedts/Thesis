import torch
import torch.optim as optim
import torch.nn.functional as F
from models import GCN
from utils import load_data, accuracy
#import train

def test(model): #similar to train, but in eval mode and on test data.
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

#print ('###in save-load.py###')
#test()
#print ('========')
#exit()
adj, features, labels, idx_train, idx_val, idx_test = load_data()

kwargs = {"dropout":0.5, "nfeat":features.shape[1], "nclass":labels.max().item()+1, "nhid":16}
model = GCN(**kwargs)
kwargs = {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4}
optimizer_sl = optim.Adam(**kwargs)

checkpoint = torch.load('model-optimised.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer_sl.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_sl = checkpoint['epoch']

#testing
model.eval()
test(model)

