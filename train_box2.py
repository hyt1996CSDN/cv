import os
import torch.nn as nn
import torch
import dataset
from torch.utils.data import DataLoader

class trainer:
    def __init__(self,net,path1,save_path,size,isCuda=True):
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
        self.path1=path1
        self.save_path=save_path
        self.isCuda=isCuda
        self.old_loss=0
        self.size=size

        # self.size2=size2
        #开启CUDA
        if self.isCuda:
            self.net = net.cuda()
        #定义损失
        self.loss1=nn.BCELoss()
        self.loss2=nn.MSELoss()

        #优化器
        self.op=torch.optim.Adam(self.net.parameters())

    def train(self):
        all_data=dataset.mtcnn_datasets(self.path1,self.size)
        batch_data2=DataLoader(all_data,batch_size=512,shuffle=True,num_workers=4)

        while True:

            loss_all = 0.0
            for i,(pixel,conf,coor,five_) in enumerate(batch_data2):

                if self.isCuda:
                    pixel=pixel.cuda()
                    conf= conf.cuda()
                    coor = coor.cuda()
                    five_=five_.cuda()
                conf_r,coor_r=self.net(pixel)
                conf_r2=conf_r.reshape(-1,1)
                #损失处理
                #置信度
                conf_index=torch.lt(conf,2)
                #标签
                conf_tag=torch.masked_select(conf,conf_index)
                #输出
                conf_r_tag=torch.masked_select(conf_r2,conf_index)

                loss_1=self.loss1(conf_r_tag,conf_tag)
                #坐标
                coor_r1=coor_r[:,0:4].reshape(-1,4)
                #五官
                coor_r2=coor_r[:,4:14].reshape(-1,10)
                #获取索引
                conf_index2=torch.gt(conf,0)

                # 获取需要的
                coor_tag2 = torch.masked_select(coor, conf_index2)
                coor_tag3 = torch.masked_select(five_, conf_index2)
                coor_r_tag2 = torch.masked_select(coor_r1, conf_index2)
                coor_r_tag3 = torch.masked_select(coor_r2, conf_index2)

                loss_2=self.loss2(coor_r_tag2,coor_tag2)
                loss_3=self.loss2(coor_r_tag3,coor_tag3)

                loss=loss_1+loss_2+loss_3

                self.op.zero_grad()
                loss.backward()
                self.op.step()
                loss_all+=loss.item()
                # if loss_all/(i+1)<old_loss:
                if (i+1)%20==0:
                    loss_all_equal=loss_all/(i+1)
                    print("loss1_总损失:{0},loss_置信度:{1},loss_坐标:{2}，平均损失{3}".format(loss.item(), loss_1.item(), loss_2.item(),loss_all_equal))
                    # old_loss=loss
                    torch.save(self.net.state_dict(), self.save_path)
                    print('save success')


