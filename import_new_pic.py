
import glob
import os
import shutil
path =r"B:\DeskTop\SRP中医体质辨识\体质辨识数据备份"
# 复制面部图像
for f in glob.glob(path+r'\data\new\*\face\*.jpg'):
    shutil.copy2(f,path+r"\origin\face")
# 复制并切分舌头
for f in glob.glob(path+r'\data\new\*\tongue\*.jpg'):
    shutil.copy2(f,path+r"\origin\tongue")