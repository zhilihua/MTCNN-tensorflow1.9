import numpy as np
import numpy.random as npr
import os

data_dir = '.'

size = 24
net = 'RNet'

with open(os.path.join(data_dir, '%s/pos_%s.txt' %(size, size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir, '%s/neg_%s.txt' %(size, size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, '%s/part_%s.txt' %(size, size)), 'r') as f:
    part = f.readlines()

with open(os.path.join(data_dir, '%s/landmark_%s_aug.txt' %(size, size)), 'r') as f:
    landmark = f.readlines()

dir_path = os.path.join(data_dir, 'imglists', "RNet")
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

#写入所有数据
with open(os.path.join(dir_path, 'train_%s_landmark.txt' % (net)), 'w') as f:
    for i in np.arange(len(pos)):
        f.write(pos[i])
    for i in np.arange(len(neg)):
        f.write(neg[i])
    for i in np.arange(len(part)):
        f.write(part[i])
    for i in np.arange(len(landmark)):
        f.write(landmark[i])
