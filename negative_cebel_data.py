import os
from PIL import Image,ImageDraw
import numpy as np
import utils
import traceback
from PIL import ImageFilter
anno_src2=r'./new_man.txt'
img_dir = r"G:\photo\mtcnn_data\img_celeba"
save_path2=r'C:\celeba4'#在固态硬盘
for face_size in [48]:
    print("gen %i image" % face_size)
    # 样本图片存储路径
    negative_image_dir = os.path.join(save_path2, str(face_size), "negative")
    if not os.path.exists(negative_image_dir):
        os.makedirs(negative_image_dir)
    # 样本描述存储路径
    negative_anno_filename = os.path.join(save_path2, str(face_size), "negative.txt")

    try:
        negative_anno_file = open(negative_anno_filename, "w")
        for i, line in enumerate(open(anno_src2)):
            if i==20000:
                break
            negative_count = 3*i
            try:
                strs = line.strip().split(" ")
                strs = list(filter(bool, strs))
                image_filename = strs[0].strip()
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    #原样本框
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    x2 = float(strs[3].strip())
                    y2 = float(strs[4].strip())
                    w = float(x2 -x1)
                    h = float(y2-y1)
                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    # 画原样本框
                    # draw_a1=ImageDraw.Draw(img)
                    # draw_a1.rectangle((x1,y1,x2,y2),outline='blue')
                    # 使正样本和部分样本数量翻倍
                    for _ in range(100):
                        if negative_count == 3*(i+1) :
                            break
                        # 让人脸中心点有少许的偏移
                        cx_ = np.random.randint(img_w*0.2,img_w*0.8)
                        cy_=np.random.randint(img_h*0.2,img_h*0.8)

                        # 部分样本
                        # x2,y2
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.2 * max(w, h)))
                        cx1_ = cx_ - side_len/2
                        cy1_ = cy_ - side_len/2
                        cx2_= cx1_+side_len
                        cy2_= cy1_+side_len

                        # 判断超出原图片没
                        if cx2_>=img_w or cy2_>=img_h or cx1_<=0 or cy1_<=0:
                            continue

                        # 画出随机样本框
                        # draw_a=ImageDraw.Draw(img)
                        # draw_a.rectangle(cr,outline='red')
                        # img.show()
                        # 计算坐标的偏移值
                        crop_box=np.array([cx1_,cy1_,cx2_,cy2_])
                        # 剪切下图片，并进行大小缩放
                        temp=[]
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size),Image.ANTIALIAS)
                        temp.append(face_resize)
                        #判断iou
                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        # print(iou)
                        if iou <0.3 :
                            for temp2 in temp:
                                negative_anno_file.write(
                                    "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                        negative_count, 0))
                                negative_anno_file.flush()
                                temp2.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                                negative_count += 1

            except Exception as e:
                traceback.print_exc()
    finally:
        negative_anno_file.close()