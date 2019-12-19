import os
from PIL import Image,ImageDraw
import numpy as np
import utils
import traceback
from PIL import ImageFilter
#路径
anno_src1=r'D:\hyt\project\mtcnn_\list_landmarks_celeba.txt'
anno_src2=r'./new_man.txt'
img_dir = r"G:\photo\mtcnn_data\img_celeba"
save_path2=r'C:\celeba4'#在固态硬盘
#开始造样本
for face_size in [48]:
    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path2, str(face_size), "positive")
    if not os.path.exists(positive_image_dir):
            os.makedirs(positive_image_dir)
    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path2, str(face_size), "positive.txt")
    try:
        positive_anno_file = open(positive_anno_filename, "w")
        for i, (line,line2) in enumerate(zip(open(anno_src2),open(anno_src1))):
            if i==2000:
                break
            positive_count = i
            try:
                # 五官
                strs2 = line2.strip().split(' ')
                strs2 = list(filter(bool, strs2))
                #框
                strs = line.strip().split(" ")
                strs = list(filter(bool, strs))
                image_filename = strs[0].strip()
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    x2 = float(strs[3].strip())
                    y2 = float(strs[4].strip())
                    w=float(x2-x1)
                    h=float(y2-y1)
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
                    for _ in range(50):
                        if positive_count == i+1:
                            break
                        # 让人脸中心点有少许的偏移
                        # w_ = np.random.randint(-w * 0.01, w * 0.01)
                        # h_ = np.random.randint(-h * 0.01, h * 0.01)
                        # cx_ = cx + w_
                        # cy_ = cy + h_
                        # 让人脸形成正方形，并且让坐标也有少许的偏离
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.2 * max(w, h)))
                        x1_ = np.max(cx - side_len / 2, 0)
                        y1_ = np.max(cy - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len
                        if x2_>=img_w or y2_>=img_h or x1_<=0 or y1_ <=0:
                            continue
                        # # 判断超出原图片没
                        # x1_=np.maximum(x1_,0)
                        # y1_=np.maximum(y1_,0)
                        # x2_=np.minimum(x2_,img_w)
                        # y2_=np.minimum(y2_,img_h)
                        crop_box = np.array([x1_, y1_, x2_, y2_])
                        # 画出随机样本框
                        # draw_a=ImageDraw.Draw(img)
                        # draw_a.rectangle((x1_,y1_,x2_,y2_),outline='red')
                        # img.show()
                        # 计算坐标的偏移值
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len
                        #五官
                        offset_left_eye_x = (float(strs2[1].strip()) - x1_) / side_len
                        offset_left_eye_y = (float(strs2[2].strip()) - y1_) / side_len
                        offset_right_eye_x = (float(strs2[3].strip()) - x1_) / side_len
                        offset_right_eye_y = (float(strs2[4].strip()) - y1_) / side_len
                        offset_nose_x = (float(strs2[5].strip()) - x1_) / side_len
                        offset_nose_y = (float(strs2[6].strip()) - y1_) / side_len
                        offset_left_mouth_x = (float(strs2[7].strip()) - x2_) / side_len
                        offset_left_mouth_y = (float(strs2[8].strip()) - y2_) / side_len
                        offset_right_mouth_x = (float(strs2[9].strip()) - x2_) / side_len
                        offset_right_mouth_y = (float(strs2[10].strip()) - y2_) / side_len
                        # 剪切下图片，并进行大小缩放
                        temp=[]
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size),Image.ANTIALIAS)
                        temp.append(face_resize)

                        #判断iou
                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        # print(iou)
                        # 正样本
                        if iou > 0.65:
                            for temp2 in temp:
                                positive_anno_file.write(
                                    "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                        positive_count, 1, offset_x1, offset_y1,
                                        offset_x2, offset_y2,offset_left_eye_x,offset_left_eye_y,offset_right_eye_x,offset_right_eye_y,offset_nose_x,offset_nose_y,offset_left_mouth_x,
                                        offset_left_mouth_y,offset_right_mouth_x,offset_right_mouth_y))
                                positive_anno_file.flush()
                                temp2.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                                positive_count += 1
            except Exception as e:
                traceback.print_exc()
    finally:
        positive_anno_file.close()
