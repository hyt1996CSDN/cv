import os
from PIL import Image,ImageDraw
import numpy as np
import traceback
import utils
star_photo=r'G:\photo\mtcnn_data\img_celeba'
star_txt=r'G:\photo\mtcnn_data\list_bbox_celeba.txt'
all_data=r'G:\photo\mtcnn_data'
five_=r'G:\photo\mtcnn_data\Anno\list_landmarks_celeba.txt'
new_yuan_path=r'G:\photo\mtcnn_data\yuan_data1.txt'
for i in [48]:
    positive_img_path=os.path.join(all_data,str(i),'positive')
    negative_img_path=os.path.join(all_data,str(i),'negative')
    part_img_path=os.path.join(all_data,str(i),'part')

    for a in [positive_img_path,negative_img_path,part_img_path]:
        if not os.path.exists(a):
            os.makedirs(a)
    #样本标签路径

    positive_data_path = os.path.join(all_data, str(i), 'positive.txt')
    negative_data_path = os.path.join(all_data, str(i), 'negative.txt')
    part_data_path = os.path.join(all_data, str(i), 'part.txt')

    #样本标签打开

    positive_data_open=open(positive_data_path,'w')
    negative_data_open = open(negative_data_path, 'w')
    part_data_open = open(part_data_path, 'w')
    positive_count = 0
    negative_count = 0
    part_count = 0
    #读取五官数据
    # five_data_read=open(five_)
    # new_yuan_data=open(new_yuan_path,'w')
    # for i2,line2 in enumerate(open(five_)):
    #     if i2<2:
    #         continue
    #     read_line=line2.strip().split(' ')
    #     read_line2=list(filter(bool,read_line))
    #     read_line2=list(map(int,read_line2[1:]))
    #     print(read_line2)
    #     x_min_dian=min(read_line2[0],read_line2[2],read_line2[4],read_line2[6],read_line2[8])
    #     x_max_dian=max(read_line2[0],read_line2[2],read_line2[4],read_line2[6],read_line2[8])
    #     y_min_dian=min(read_line2[1],read_line2[3],read_line2[5],read_line2[7],read_line2[9])
    #     y_max_dian = max(read_line2[1], read_line2[3], read_line2[5], read_line2[7], read_line2[9])
    #     new_yuan_data.write(
    #         "{0} {1} {2} {3}\n".format(x_min_dian,y_min_dian,x_max_dian,y_max_dian)
    #     )
    #原样本更新（根据五官）
    read_yuan3=[]
    for j2,line in enumerate(open(new_yuan_path)):
        read_yuan1=list(line.strip().split(' '))
        read_yuan3.append(read_yuan1)

    for j ,line in enumerate(open(star_txt)):
        if j <2:
            continue
        a=line.strip().split(' ')
        b=list(filter(bool,a))
        print(b)
        photo_name=b[0].strip()
        photo_path=os.path.join(star_photo,str(photo_name))

        with Image.open(photo_path) as img:
            x1=float(b[1].strip())
            y1=float(b[2].strip())
            w=float(b[3].strip())
            h=float(b[4].strip())
            x2=float(x1+w)
            y2=float(y1+h)
            z_x=w/2+x1
            z_y=h/2+y1
            # yuan_x1=float(read_yuan3[j][0].strip())
            # yuan_y1=float(read_yuan3[j][1].strip())
            # yuan_x2=float(read_yuan3[j][2].strip())
            # yuan_y2=float(read_yuan3[j][3].strip())
            # cha_x1=yuan_x1-0.7*(yuan_x1-x1)
            # cha_y1=yuan_y1-0.8*(yuan_y1-y1)
            # cha_x2=yuan_x2+0.7*(x2-yuan_x2)
            # cha_y2=yuan_y2+0.8*(y2-yuan_y2)
            # new_yuan_w=yuan_x2-yuan_x1
            # new_yuan_h=yuan_y2-yuan_y1
            # yuan_x11=yuan_x1-0.3*new_yuan_w
            # yuan_y11=yuan_y1-1*new_yuan_w
            # yuan_x22=yuan_x2+0.8*new_yuan_h
            # yuan_y22=yuan_y2+0.8*new_yuan_h
            yuan_size=np.array([[x1,y1,x2,y2]])
            draw_1=ImageDraw.Draw(img)
            draw_1.rectangle((x1,y1,x2,y2),outline='red')
            # draw_1.rectangle((cha_x1,cha_y1,cha_x2,cha_y2),outline='blue')
            img.show()
            if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                continue
            for _ in range(5):
                n_x=z_x+np.random.randint(-w*0.2,w*0.2)
                n_y=z_y+np.random.randint(-h*0.2,h*0.2)
                side_len=np.random.randint(int(0.8*min(w,h)),np.ceil(1.25*max(w,h)))
                #偏移后的2点坐标
                n_x1=np.max(n_x-side_len/2,0)
                n_y1=np.max(n_y-side_len/2,0)
                n_x2=n_x1+side_len
                n_y2=n_y1+side_len
                #偏移量
                offset_x1=(x1-n_x1)/side_len
                offset_y1=(y1-n_y1)/side_len
                offset_x2=(x2-n_x2)/side_len
                offset_y2=(y2-n_y2)/side_len
                crop_size=np.array([n_x1,n_y1,n_x2,n_y2])

                #剪切并缩放
                img_crop=img.crop(crop_size)
                img_resize=img_crop.resize((int(i),int(i)))
                iou_1=utils.iou(crop_size, yuan_size)
                print(iou_1)
                if iou_1>0.4:
                    positive_data_open.write(
                        "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                            positive_count, 1, offset_x1, offset_y1, offset_x2,
                            offset_y2))
                    positive_data_open.flush()
                    img_resize.save(os.path.join(positive_img_path, "{0}.jpg".format(positive_count)))
                    positive_count += 1
                if iou_1>0.3:
                    part_data_open.write(
                        "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                            part_count, 2, offset_x1, offset_y1, offset_x2,
                            offset_y2))
                    part_data_open.flush()
                    img_resize.save(os.path.join(part_img_path, "{0}.jpg".format(part_count)))
                    part_count += 1
                if iou_1<0:
                    negative_data_open.write(
                        "negative/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                            negative_count, 0, 0, 0, 0,
                            0))
                    negative_data_open.flush()
                    img_resize.save(os.path.join(negative_img_path, "{0}.jpg".format(negative_count)))
                    negative_count += 1








