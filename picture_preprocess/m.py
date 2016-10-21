from PIL import Image
import dlib
from white_balance import to_pil,stretch,from_pil
import os
# 图片处理

# 读取数据库
import pymysql
conn = pymysql.connect(host='localhost', port=3306,user='root',passwd='password',db='my_db',charset='UTF8')
cur = conn.cursor()
cur.execute("select id,tongue_photo,face_photo from sample")
for id,tongue,face in cur:
    local_path = r"C:\offline\体质辨识数据备份"# 本地存储地址
    tongue = tongue.replace(r"/home/wenserver",local_path)
    face = face.replace(r"/home/wenserver", local_path)
    # 处理face
    if not os.path.exists(face):
        continue
    img = Image.open(face)
    after = to_pil(stretch(from_pil(img)))
    img_name = os.path.split(face)[1]
    after.save(os.path.join(local_path,'stretch','face',img_name))
    #处理tongue
    if not os.path.exists(tongue):
        continue
    img = Image.open(tongue)
    after = to_pil(stretch(from_pil(img)))
    img_name = os.path.split(face)[1]
    after.save(os.path.join(local_path, 'stretch', 'tongue', img_name))
