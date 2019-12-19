from torch.utils.data import Dataset
import os
from PIL import Image,ImageFilter
import torch
import torchvision
import numpy as np
class mtcnn_datasets(Dataset):
    def __init__(self,path,a):
        self.path=path
        self.box=[]
        self.box.extend(open(os.path.join(self.path,'positive.txt')).readlines())
        print(len(self.box))
        self.box.extend(open(os.path.join(self.path,'negative.txt')).readlines())
        print(len(self.box))
        self.box.extend(open(os.path.join(self.path,'part.txt')).readlines())
        print(len(self.box))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((int(a/2), int(a/2))),
            torchvision.transforms.Resize((a, a )),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transform2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((a, a)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def __len__(self):
        return len(self.box)

    def __getitem__(self, index):
        a1=self.box[index].split()
        img_path=os.path.join(self.path,a1[0])
        img_da=Image.open(img_path)
        # i = np.random.randint(0, 2)
        # if i==0:
        #     img_data1=self.transform(img_da.filter(ImageFilter.BLUR))
        # else:
        #
        # if i==0:
        #     img_data1=self.transform2(img_da)
        # else:
        #     img_data1=self.transform(img_da)
        img_data1 = self.transform2(img_da)
        img_data2=torch.tensor([float(a1[1])])
        img_data3=torch.tensor([float(a1[2]),float(a1[3]),float(a1[4]),float(a1[5])])
        img_data4=torch.tensor([float(a1[6]),float(a1[7]),float(a1[8]),float(a1[9]),float(a1[10]),float(a1[11]),float(a1[12]),float(a1[13]),float(a1[14]),float(a1[15])])
        return img_data1,img_data2,img_data3,img_data4

if __name__ == '__main__':
    mtc=mtcnn_datasets(r'G:/photo/mtcnn_data/celeba/12',12)
    a=mtc[1]