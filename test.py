# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle
# f = open(r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin\tongue_color_hist",'rb')
# tongue = pickle.load(f)
# f = open(r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin\face_color_hist",'rb')
# face = pickle.load(f)
# f = open(r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin\face_block_color_hist",'rb')
# block = pickle.load(f)
#
# c1,c2,c3 = 0,0
# for (k,v) in tongue.items():
#     if k in face.keys():
#         c1+=1
#         if k in block.keys():
#             c2+=1
# print(c1,c2)

# import numpy as np
# a = np.array((True,True,True,False))
# b = np.array((1,1,3,4))
# print(a*1*b)

# import glob
# import os
# from skimage import io
# for  f in glob.glob(os.path.join(r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin\cutted_tongue", "*.jpg")):
#     # 计算特征
#     img = io.imread(f)
#     # 排除太小的图片
#     if (img.shape[0] * img.shape[1] > 20000 and img.shape[0] * img.shape[1] < 40000):
#         io.imshow(img)
#         io.show()

# import os,glob
# face,tongue = [],[]
# for f in glob.glob(r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin\face\*.jpg"):
#     fname = os.path.split(f)[1].split('.')[0]
#     if len(fname.split('_')) > 1:
#         face.append(fname.split('_')[0])
# for f in glob.glob(r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin\cutted_tongue\*.jpg"):
#     fname = os.path.split(f)[1].split('.')[0]
#     if len(fname.split('_')) > 1:
#         tongue.append(fname.split('_')[0])
# print(len(face),len(tongue))
# num =0
# for f in face:
#     if f in tongue:
#         num+=1
# print(num)

# from sklearn.preprocessing import scale
# import numpy as np
# a = np.array(((1,2),(4,5),(7,8)),dtype=np.float)
# print(a)
# print(a.shape)
# b = np.sum(a,axis=1).reshape((a.shape[0],1))
# print(b)
# print(a/b)


# import numpy as np
# from skimage import io
# from sklearn.model_selection import KFold
#
# a = [2,2,3,4,3,2,3,2,1,10]
# kf = KFold(n_splits=10)
# for train,test in kf.split(a):
#     print(train,test)

import numpy as np
a = np.random.random(4)*8
print(a)
print(a.astype(np.int8))