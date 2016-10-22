import glob
import os
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
lists = []
path = r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\4466"
for f in glob.glob(path+r"\face\*.jpg"):
    fname = os.path.split(f)[1].split('.')[0]
    if len(fname.split('_')) > 1:
        # ID = fname.split('_')[0]
        # labels = fname.split("_")[1:]
        # lists.append([ID,labels])
        lists.append(fname.split('_'))
print(lists)
f = open(path+r"\labels",'wb')
pickle.dump(lists,f)