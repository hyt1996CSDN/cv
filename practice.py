from PIL import Image,ImageDraw
import numpy as np
import torch
import torchvision
# a=[1,2]
# b=[2,3]
# c=[]
# c.extend((a,b))
# print(c)
#//////////////////////////////////////
# import net
# import torch
# import os
from PIL import ImageDraw,Image,ImageFilter
# import torchvision
# for i in range(50000):
#     a=Image.open(os.path.join(r'C:\celeba\48\positive','{0}.jpg'.format(i)))
#     a2=a.filter(ImageFilter.BLUR)
#     a2.save(os.path.join(r'C:\celeba\48\positive','{0}.jpg'.format(i)))
# file1_path=r'G:\photo\mtcnn_data\Anno\list_landmarks_celeba.txt'
# file2_path=r'G:\photo\mtcnn_data\list_bbox_celeba.txt'
# file_write_path=r'./new_man2.txt'
# for i,(tta,ttb) in enumerate(zip(open(file1_path),open(file2_path))):
#     if i<2:
#         continue
#     str=tta.strip().split()
#     str2=ttb.strip().split()
#     new_str=list(filter(bool,str))
#     print(str)
#     new_str2 = list(filter(bool, str2))
#     #五官框
#     min_x=min(float(new_str[1]),float(new_str[3]),float(new_str[5]),float(new_str[7]),float(new_str[9]))
#     min_y=min(float(new_str[2]),float(new_str[4]),float(new_str[6]),float(new_str[8]),float(new_str[10]))
#     max_x=max(float(new_str[1]),float(new_str[3]),float(new_str[5]),float(new_str[7]),float(new_str[9]))
#     max_y=max(float(new_str[2]),float(new_str[4]),float(new_str[6]),float(new_str[8]),float(new_str[10]))
#
#     #求五官框与原框的距离
#     new_w1=abs(min_x-float(str2[1]))
#     new_h1=abs(min_y-float(str2[2]))
#     new_w2=abs(float(str2[3])+float(str2[1])-max_x)
#     new_h2=abs(float(str2[4])+float(str2[2])-max_y)
#
#     write_file=open(file_write_path,'a')
#     write_file.write('{0} {1} {2} {3} {4}\n'.format(str[0],int(min_x-new_w1*0.8),int(min_y-new_h1*1),int(max_x+new_w2*0.8),int(max_y+new_h2*0.7)))
a=Image.open(r'D:\hyt\project\mtcnn_\photo_arc_loss\2_1.jpg')
a2=  torchvision.transforms.Resize(200)
a3=a2(a)
w,h=a3.size
max_wh=max(w,h)
a22=torchvision.transforms.CenterCrop(max_wh)
a4=a22(a3)
a4.show()