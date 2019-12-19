import torch
import net
import utils
import numpy as np
from PIL import Image, ImageDraw
import torchvision
import time

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
     ]
)


class tracker:
    def __init__(self, p_net='./state_p.pt', r_net='./state_r.pt', o_net='./state_o.pt', isCuda=True):
        # 创建网络
        self.p_net = net.Net_p()
        self.r_net = net.Net_R()
        self.o_net = net.Net2_O()
        self.isCuda = isCuda
        # 给网络加载训练好的参数
        self.p_net.load_state_dict(torch.load(p_net))
        self.r_net.load_state_dict(torch.load(r_net))
        self.o_net.load_state_dict(torch.load(o_net))
        # 变为CUDA网络
        if self.isCuda:
            self.p_net.cuda()
            self.r_net.cuda()
            self.o_net.cuda()
        # eval()因为测试网络与训练不同
        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

    def p_tracker(self, img):
        # 获取照片信息
        time_13 = time.time()
        p_box_end = []
        p_box2 = []
        img_w, img_h = img.size
        img_wh = min(img_w, img_h)
        # 对图像金字塔判断
        scale = 1
        img_tensor = transform(img)
        time_14 = time.time()
        while img_wh > 12:
            img_tensor.unsqueeze_(0)
            if self.isCuda:
                img_tensor = img_tensor.cuda()
            # 放入p网络测试
            with torch.no_grad():
                conf, coor = self.p_net(img_tensor)  # conf置信度shape[1,1,w,h]，coor偏移[1,4,w,h]
            conf = conf.cpu().detach()
            coor = coor.cpu().detach()

            conf = conf[0][0]  # [h,w]
            coor = coor[0]  # [4,h,w]

            # coor索引获取
            index = torch.nonzero(torch.gt(conf, 0.6))  # [[1,2cupu....],[3,4....]]
            # h,w
            r_in = self.yuan_kuang(index, conf[index[:, 0], index[:, 1]], scale, coor, 2, 12)  # [[4偏移，1置信度]]
            scale *= 0.7
            w = int(img_w * scale)
            h = int(img_h * scale)
            img_wh = min(w, h)
            img1 = img.resize((w, h))
            img_tensor = transform(img1)
            p_box2.extend(utils.nsm(np.array(r_in), 0.5))  # nsm返回[[  ]] 在NMS留下的所有框
        time_15 = time.time()

        p_box3 = utils.convert_to_square(np.array(p_box2))
        time_16 = time.time()
        for p_box3_one in p_box3:
            x1 = p_box3_one[0]
            y1 = p_box3_one[1]
            x2 = p_box3_one[2]
            y2 = p_box3_one[3]
            img_1 = img.crop((x1, y1, x2, y2))
            # a=ImageDraw.ImageDraw(img)
            # a.rectangle((x1,y1,x2,y2),outline='green')#bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb                      画框
            img_1 = img_1.resize((24, 24))
            img_1 = transform(img_1)
            p_box_end.append(img_1)
        p_box_end = torch.stack(p_box_end)
        time_17 = time.time()
        print(time_14-time_13,time_15-time_14,time_16-time_15,time_17-time_16)
        if self.isCuda:
            p_box_end = p_box_end.cuda()
        return p_box_end, p_box3
        # 输出为24*24像素值，24*24o网络输入坐标，P网络输出的坐标

    def r_tracker(self, p_box, p_box2):
        r_box = []
        r_box_end = []
        with torch.no_grad():
            r_conf, r_coor = self.r_net(p_box)

        r_conf = r_conf.cpu().detach().numpy()
        r_coor = r_coor.cpu().detach().numpy()
        # print(r_conf)
        index_1, index_2 = np.where(r_conf > 0.6)
        for index_ in index_1:
            r_box.append(self.yuan_kuang2(index_, r_conf[index_], r_coor, p_box2,0))

        # 在原图上扣正方形图用于传入o网络
        # print()
        r_box2 = utils.nsm(np.stack(r_box), 0.6)
        r_box3 = utils.convert_to_square(np.stack(r_box2))
        for r_box3_one in r_box3:
            x1 = r_box3_one[0]
            y1 = r_box3_one[1]
            x2 = r_box3_one[2]
            y2 = r_box3_one[3]
            img_1 = img.crop((x1, y1, x2, y2))
            a = ImageDraw.ImageDraw(img)
            # a.rectangle((x1, y1, x2, y2),
            #             outline='blue')  # bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
            img_1 = img_1.resize((48, 48))
            img_1 = transform(img_1)
            r_box_end.append(img_1)
        p_box_end = torch.stack(r_box_end)
        if self.isCuda:
            r_box_end = p_box_end.cuda()
        return r_box_end, r_box3

    def o_tracker(self, r_box, r_box2):
        o_box = []
        with torch.no_grad():
            o_conf, o_coor = self.o_net(r_box)
        o_conf = o_conf.cpu().data.numpy()
        o_coor = o_coor.cpu().data.numpy()
        index1, index2 = np.where(o_conf > 0.9)
        for index_ in index1:
            o_box.append(self.yuan_kuang2(index_, o_conf[index_], o_coor, r_box2, 1))
        return utils.nsm(np.array(o_box), 0.6, c=1)

    def yuan_kuang(self, index_, conf, scale, coor, stride, side_len):
        # 原建议框
        yuan_1 = (index_ * stride).float() / scale
        yuan_2 = (index_ * stride + side_len).float() / scale
        # 原建议框大
        yuan_conf = coor[:, index_[:, 0], index_[:, 1]] * (side_len / scale)
        # 人脸框
        conf_x1 = yuan_1[:, 1] + yuan_conf[0]
        conf_y1 = yuan_1[:, 0] + yuan_conf[1]
        conf_x2 = yuan_2[:, 1] + yuan_conf[2]
        conf_y2 = yuan_2[:, 0] + yuan_conf[3]

        t = torch.cat(
            [conf_x1.unsqueeze(1), conf_y1.unsqueeze(1), conf_x2.unsqueeze(1), conf_y2.unsqueeze(1), conf.unsqueeze(1)],
            dim=1)
        return t  # [[4偏移，1置信度]]

    def yuan_kuang2(self, index_, conf, coor, p_box,a):
        _box = p_box[index_]
        yuan_x1 = int(_box[0])
        yuan_y1 = int(_box[1])
        yuan_x2 = int(_box[2])
        yuan_y2 = int(_box[3])
        yuan_w = yuan_x2 - yuan_x1
        yuan_h = yuan_y2 - yuan_y1
        conf_x1 = yuan_x1 + coor[index_][0] * yuan_w
        conf_y1 = yuan_y1 + coor[index_][1] * yuan_h
        conf_x2 = yuan_x2 + coor[index_][2] * yuan_w
        conf_y2 = yuan_y2 + coor[index_][3] * yuan_h
        if a==1:
            x1=yuan_x1 + coor[index_][4] * yuan_w
            y1=yuan_y1 + coor[index_][5] * yuan_h
            x2 = yuan_x1 + coor[index_][6] * yuan_w
            y2 = yuan_y1 + coor[index_][7] * yuan_h
            x3 = yuan_x1 + coor[index_][8] * yuan_w
            y3 = yuan_y1 + coor[index_][9] * yuan_h
            x4 = yuan_x2 + coor[index_][10] * yuan_w
            y4 = yuan_y2 + coor[index_][11] * yuan_h
            x5 = yuan_x2 + coor[index_][12] * yuan_w
            y5 = yuan_y2 + coor[index_][13] * yuan_h
            return [conf_x1, conf_y1, conf_x2, conf_y2, conf,x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]  # [[4偏移，1置信度]]
        else:
            return [conf_x1, conf_y1, conf_x2, conf_y2, conf]  # [[4偏移，1置信度]]

    def star(self, img):
        p_t_sta = time.time()
        p_box1, p_box2 = self.p_tracker(img)
        p_t_end = time.time()
        r_box1, r_box2 = self.r_tracker(p_box1, p_box2)
        r_t_end = time.time()
        box3 = self.o_tracker(r_box1, r_box2)
        o_t_end = time.time()
        print(
            "p_time:{},r_time:{},o_time:{},all_time:{}".format(p_t_end - p_t_sta, r_t_end - p_t_end, o_t_end - r_t_end,
                                                               o_t_end - p_t_sta))
        return box3  # 输出为[[4偏移，1置信]]


if __name__ == '__main__':
    img_path = r'D:\hyt\project\mtcnn_\photo_arc_loss\2_1.jpg'
    time_12 = time.time()
    tra1 = tracker('./state_p_60.pt', './state_r_60.pt', './o_five4.pt')
    time_13 = time.time()
    img = Image.open(img_path)
    time_14 = time.time()
    final_box = tra1.star(img)
    time_15 = time.time()
    print(time_13-time_12,time_14-time_13,time_15-time_14)
    draw = ImageDraw.ImageDraw(img)
    for final_box_one in final_box:
        draw_x1 = final_box_one[0]
        draw_y1 = final_box_one[1]
        draw_x2 = final_box_one[2]
        draw_y2 = final_box_one[3]
        draw_el_x1 = final_box_one[5]
        draw_el_y1 = final_box_one[6]
        draw_er_x1 = final_box_one[7]
        draw_er_y1 = final_box_one[8]
        draw_n_x1 = final_box_one[9]
        draw_n_y1 = final_box_one[10]
        draw_ml_x1 = final_box_one[11]
        draw_ml_y1 = final_box_one[12]
        draw_mr_x1 = final_box_one[13]
        draw_mr_y1 = final_box_one[14]
        draw.rectangle((draw_x1, draw_y1, draw_x2, draw_y2), outline='red',
                       width=3)  # ///////////////////////////////////////////////////////////////////////
        draw.point((draw_el_x1,draw_el_y1),fill='red')
        draw.point((draw_er_x1, draw_er_y1), fill='red')
        draw.point((draw_n_x1, draw_n_y1), fill='red')
        draw.point((draw_ml_x1, draw_ml_y1), fill='red')
        draw.point((draw_mr_x1, draw_mr_y1), fill='red')
    img.show()