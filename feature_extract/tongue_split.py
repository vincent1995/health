# -*- coding: utf-8 -*-
'''
Created on 2016年7月22日
@author: Administrator
'''
def get_landmarks(im,path,dirpath):
    import dlib
    from skimage import io
    import os
    predictor_path = "module.dat"  # 获取训练模型
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    rects=detector(im,1)
    flag=False
    if(len(rects)>0):
        flag=True
        postion=predictor(im,rects[0])
        p1=postion.part(8)
        p2=postion.part(48)
        p3=postion.part(54)
        p4=postion.part(33)
        left=int(p2.x)
        top=int(p4.y)
        right=int(p3.x)
        down=int(p1.y+5)
        im=im[top:down,left:right]
        print(im.shape)
        s_path=os.path.split(path)[1]
        t_path=os.path.join(dirpath+r"\cutted_tongue",s_path)
        io.imsave(t_path,im)
    return flag

def split():
    dirpath = "B:\DeskTop\SRP中医体质辨识\体质辨识数据备份\origin"
    import glob
    from skimage import io
    import os
    num=0;
    for i,f in enumerate(glob.glob(dirpath+r"\tongue\*.jpg")):
        s_path = os.path.split(f)[1]
        if(os.path.exists(os.path.join(dirpath,"cutted_tongue",s_path))):
            continue
        import imghdr
        pic_type = imghdr.what(f)
        if pic_type:
            img=io.imread(f)
            flag=get_landmarks(img,f,dirpath)
        else:
            flag = False
        if flag:
            num=num+1;
            print(i,f+" is matching")
        else:
            print(i,f+" is no matching face")
    print("the number of face is {}".format(num))

split()