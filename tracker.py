import torch
import net
import utils
import numpy as np
from PIL import Image,ImageDraw
import torchvision
import time
transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
     ]
)
class tracker:
    def __init__(self,p_net='./state_p.pt',r_net='./state_r.pt',o_net='./state_o.pt',isCuda=True):
        #创建网络
        self.p_net=net.Net_p()
        self.r_net=net.Net_R()
        self.o_net=net.Net_O()
        self.isCuda=isCuda
        #给网络加载训练好的参数
        self.p_net.load_state_dict(torch.load(p_net))
        self.r_net.load_state_dict(torch.load(r_net))
        self.o_net.load_state_dict(torch.load(o_net))
        #变为CUDA网络
        if self.isCuda:
            self.p_net.cuda()
            self.r_net.cuda()
            self.o_net.cuda()
        #eval()因为测试网络与训练不同
        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

    def p_tracker(self,img):
        # 获取照片信息
        p_box_end = []
        p_box2=[]
        p_box = []
        img_w,img_h=img.size
        img_wh=min(img_w,img_h)
        #对图像金字塔判断
        scale=1
        img_tensor = transform(img)
        while img_wh>12:
            img_tensor.unsqueeze_(0)
            if self.isCuda:
                img_tensor=img_tensor.cuda()
            #放入p网络测试
            with torch.no_grad():
                conf,coor=self.p_net(img_tensor)#conf置信度shape[1,1,w,h]，coor偏移[1,4,w,h]
            conf=conf.cpu().detach()
            coor=coor.cpu().detach()
            conf=conf[0][0]#[h,w]
            coor=coor[0]#[4,h,w]
            #coor索引获取
            index=torch.nonzero(torch.gt(conf,0.6))#[[1,2cupu....],[3,4....]]#////////////////////////////////
            for index_ in index:  # h,w
                p_box.append(self.yuan_kuang(index_,conf[index_[0],index_[1]],scale,coor,2,12)) #[[4偏移，1置信度]]
            scale*=0.7
            w=int(img_w*scale)
            h=int(img_h*scale)

            img_wh=min(w,h)
            img1=img.resize((w,h))
            img_tensor = transform(img1)

            p_box2.extend(utils.nsm(np.array(p_box),0.5))#nsm返回[[  ]] 在NMS留下的所有框#////////////////////////////////
        p_box3=utils.convert_to_square(np.array(p_box2))#[[4,1置信度]]   p的正方形原型
        # print(p_box2)
        for p_box3_one in p_box3:
            x1=p_box3_one[0]
            y1=p_box3_one[1]
            x2=p_box3_one[2]
            y2=p_box3_one[3]
            img_1=img.crop((x1,y1,x2,y2))
            # a=ImageDraw.ImageDraw(img)
            # a.rectangle((x1,y1,x2,y2),outline='green')

            img_1=img_1.resize((24,24))
            img_1=transform(img_1)

            p_box_end.append(img_1)
        p_box_end=torch.stack(p_box_end)

        if self.isCuda:
            p_box_end=p_box_end.cuda()
        return p_box_end,p_box3
        #输出为24*24像素值，24*24o网络输入坐标，P网络输出的坐标

    def r_tracker(self,p_box,p_box2):
        r_box=[]
        r_box_end=[]
        with torch.no_grad():
            r_conf,r_coor=self.r_net(p_box)

        r_conf=r_conf.cpu().detach().numpy()
        r_coor=r_coor.cpu().detach().numpy()
        # print(r_conf)
        index_1,index_2=np.where(r_conf>0.6)#////////////////////////////////
        for index_ in index_1:
            r_box.append(self.yuan_kuang2(index_,r_conf[index_],r_coor,p_box2))

        #在原图上扣正方形图用于传入o网络
        # print()
        r_box2=utils.nsm(np.stack(r_box),0.5)#////////////////////////////////
        r_box3=utils.convert_to_square(np.stack(r_box2))
        for r_box3_one in r_box3:
            x1=r_box3_one[0]
            y1=r_box3_one[1]
            x2=r_box3_one[2]
            y2=r_box3_one[3]
            img_1=img.crop((x1,y1,x2,y2))
            # a = ImageDraw.ImageDraw(img)
            # a.rectangle((x1, y1, x2, y2),outline='blue')
            img_1=img_1.resize((48,48))
            img_1=transform(img_1)
            r_box_end.append(img_1)
        p_box_end=torch.stack(r_box_end)
        if self.isCuda:
            r_box_end=p_box_end.cuda()
        return r_box_end,r_box3

    def o_tracker(self,r_box,r_box2):
        o_box=[]
        with torch.no_grad():
            o_conf,o_coor=self.o_net(r_box)
        o_conf=o_conf.cpu().data.numpy()
        o_coor=o_coor.cpu().data.numpy()
        index1,index2=np.where(o_conf>0.97)#////////////////////////////////
        for index_ in  index1:
            o_box.append(self.yuan_kuang2(index_,o_conf[index_],o_coor,r_box2))

        return utils.nsm(np.array(o_box),0.6,c=1)#////////////////////////////////


    def yuan_kuang(self,index_,conf,scale,coor,stride,side_len):
        # 建议框
        yuan_x1 =float(index_[1] * stride)/scale
        yuan_y1 =float(index_[0] * stride)/scale
        yuan_x2 =float(index_[1] * stride + side_len)/scale
        yuan_y2 =float(index_[0] * stride + side_len)/scale
        yuan_w = yuan_x2 - yuan_x1
        yuan_h = yuan_y2 - yuan_y1
        conf_x1 = yuan_x1 + coor[0, index_[0], index_[1]] * yuan_w
        conf_y1 = yuan_y1 + coor[1, index_[0], index_[1]] * yuan_h
        conf_x2 = yuan_x2+ coor[2, index_[0], index_[1]] * yuan_w
        conf_y2 = yuan_y2 + coor[3, index_[0], index_[1]] * yuan_h
        return [conf_x1, conf_y1, conf_x2, conf_y2, conf  ]# [[4偏移，1置信度]]

    def yuan_kuang2(self,index_,conf,coor,p_box):
        _box=p_box[index_]
        yuan_x1 =int(_box[0])
        yuan_y1 =int(_box[1])
        yuan_x2 =int(_box[2])
        yuan_y2 =int(_box[3])
        yuan_w = yuan_x2 - yuan_x1
        yuan_h = yuan_y2 - yuan_y1
        conf_x1 = yuan_x1 + coor[index_][0] * yuan_w
        conf_y1 = yuan_y1 + coor[index_][1] * yuan_h
        conf_x2 = yuan_x2 + coor[index_][2] * yuan_w
        conf_y2 = yuan_y2 + coor[index_][3] * yuan_h

        return [conf_x1, conf_y1, conf_x2, conf_y2, conf]  # [[4偏移，1置信度]]

    def star(self,img):
        p_t_sta=time.time()
        p_box1,p_box2=self.p_tracker(img)
        p_t_end=time.time()
        r_box1,r_box2=self.r_tracker(p_box1,p_box2)
        r_t_end=time.time()
        box3=self.o_tracker(r_box1,r_box2)
        o_t_end=time.time()
        print("p_time:{},r_time:{},o_time:{},all_time:{}".format(p_t_end-p_t_sta,r_t_end-p_t_end,o_t_end-r_t_end,o_t_end-p_t_sta))
        return box3      #输出为[[4偏移，1置信]]

if __name__ == '__main__':
    img_path='./photo/1.jpg'
    tra1=tracker('./state_p_new_50w.pt','./state_r_new_50w.pt','./state_o_new_50w.pt')
    img=Image.open(img_path)
    final_box=tra1.star(img)
    draw=ImageDraw.ImageDraw(img)
    for final_box_one in  final_box:
        draw_x1=final_box_one[0]
        draw_y1=final_box_one[1]
        draw_x2=final_box_one[2]
        draw_y2=final_box_one[3]

        draw.rectangle((draw_x1,draw_y1,draw_x2,draw_y2),outline='red')
    img.show()





