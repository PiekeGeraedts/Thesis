'''
##################################################This code was taken from train_SPSA.py on 10-12-2020##################################################
grads=0
cnt = 0
maximum = 0
for edge in zerograd_edges:
    if edge[0] == edge[1]:
        continue
    cnt+=1
    combined = torch.cat((torch.where(adj_indices[0] == edge[0])[0], torch.where(adj_indices[1] == edge[1])[0]))
    uniques, counts = combined.unique(return_counts=True)
    grads += torch.abs(model.edge_weights.grad[uniques[counts > 1]])
    if (torch.abs(model.edge_weights.grad[uniques[counts > 1]]) > maximum):
        maximum = torch.abs(model.edge_weights.grad[uniques[counts > 1]])

nzgrads = 0
for edge in nzgrad_edges:
    combined = torch.cat((torch.where(adj_indices[0] == edge[0])[0], torch.where(adj_indices[1] == edge[1])[0]))
    uniques, counts = combined.unique(return_counts=True)
    nzgrads += torch.abs(model.edge_weights.grad[uniques[counts > 1]])

sum_grads = torch.sum(torch.abs(model.edge_weights.grad))

print ('estimate of number of (zero) gradient edges:', cnt)
print ('sum of (zero) gradients:', grads)
print ('average of (zero) gradients:', grads/cnt)
print ('maximum of (zero) gradient:', maximum)
print ('percentage of total:', grads/sum_grads)
print ('=====================================')
print ('sum of (non zero) gradients:', nzgrads)
print ('average of (non zero) gradients:', nzgrads/nnz_edges)
print ('percentage of total:', nzgrads/sum_grads)
#print (model.edge_weights.grad[0])  #edge weight for p_{00}
print (sum_grads)
################################################
#make idx for edge weights of the test set
idx_test_vals = []
for node in idx_test:
    idx_test_vals.append(torch.where(adj_indices[0] == node)[0])
    #idx_test_vals.append(torch.where(adj_indices[1] == node)[0])
idx_test_vals = [idx for idcs in idx_test_vals for idx in idcs.tolist()]

# make idx for edge weights of the train set
idx_train_vals = []
for node in idx_train:
    idx_train_vals.append(torch.where(adj_indices[0] == node)[0])
    #idx_train_vals.append(torch.where(adj_indices[1] == node)[0])
idx_train_vals = [idx for idcs in idx_train_vals for idx in idcs.tolist()]

# make idx for edge weights of the val set
idx_val_vals = []
for node in idx_val:
    idx_val_vals.append(torch.where(adj_indices[0] == node)[0])
    #idx_val_vals.append(torch.where(adj_indices[1] == node)[0])
idx_val_vals = [idx for idcs in idx_val_vals for idx in idcs.tolist()]

#determine one step neighbourhood of the training set
step1_nghd = set()
for i in idx_train:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step1_nghd.add(int(node))
print ('one step size:', len(step1_nghd))

###two step neighbourhood
step2_nghd = set()
for i in step1_nghd:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step2_nghd.add(int(node))
print ('two step size:', len(step2_nghd))

###three step neighbourhood
step3_nghd = set()
for i in step2_nghd:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step3_nghd.add(int(node))
print ('three step size:', len(step3_nghd))

###four step neighbourhood
step4_nghd = set()
for i in step3_nghd:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step4_nghd.add(int(node))
print ('four step size:', len(step4_nghd))

###five step neighbourhood
step5_nghd = set()
for i in step4_nghd:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step5_nghd.add(int(node))
print ('five step size:', len(step5_nghd))

###six step neighbourhood
step6_nghd = set()
for i in step5_nghd:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step6_nghd.add(int(node))
print ('six step size:', len(step6_nghd))

###seven step neighbourhood
step7_nghd = set()
for i in step6_nghd:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step7_nghd.add(int(node))
print ('seven step size:', len(step7_nghd))

###eight step neighbourhood
step8_nghd = set()
for i in step7_nghd:
    for node in adj_indices[1][torch.where(adj_indices[0] == i)]:
        step8_nghd.add(int(node))
print ('eight step size:', len(step8_nghd))

complement_nghd = []
for i in range(adj_size[0]):
    if (i not in step1_nghd):
        complement_nghd.append(i)
print ('1 step nghd size=', len(step1_nghd))
print ('2+ step nghd size=', len(complement_nghd))

nz_edges = 0
nnz_edges = 0
zerograd_edges = []
nzgrad_edges = []
for i in complement_nghd:
    for j in adj_indices[1][torch.where(adj_indices[0] == i)].tolist():
        if j in complement_nghd:
            zerograd_edges.append([i,j])
            nz_edges += 1
for i in step1_nghd:
    for j in adj_indices[1][torch.where(adj_indices[0] == i)].tolist():
        nzgrad_edges.append([i,j])
        nnz_edges += 1
print ('number of edges in 1 step nghd=', nnz_edges)
print ('number of edges in 2+ step nghd=', nz_edges)
##################################################This code was taken from train_SPSA.py on 10-12-2020##################################################
'''


'''
##################################################This code was taken from models.py on 10-12-2020##################################################
def forward2(self, x, indices, values, size):
    h = F.relu(self.gc1(x, indices, self.edge_weights, size))
    h = F.dropout(h, self.dropout, training=self.training)
    z = self.gc2(h, indices, self.edge_weights, size)
    #F.log_softmax(z, dim=1)
    #-- we do not give back the normalised z.
    return x, h, z

def forward4(self, x, indices, values, size):
    h1 = F.relu(self.gc1(x, indices, self.edge_weights, size))
    h1 = F.dropout(h1, self.dropout, training=self.training)
    h2 = F.relu(self.gc2(h1, indices, self.edge_weights, size))
    h2 = F.dropout(h2, self.dropout, training=self.training)
    h3 = F.relu(self.gc3(h2, indices, self.edge_weights, size))
    h3 = F.dropout(h3, self.dropout, training=self.training)
    z = self.gc4(h3, indices, self.edge_weights, size)
    #F.log_softmax(z, dim=1)
    #-- we do not give back the normalised z.
    return x, h1, h2, h3, z
##################################################This code was taken from models.py on 10-12-2020##################################################
'''






