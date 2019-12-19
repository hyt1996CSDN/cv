import train_box1
import train_box2
import net
if __name__ == '__main__':
    O_net=net.Net2_O()
    train_O=train_box2.trainer(O_net,r'C:/celeba2/48','./o_five4.pt',48)#背景多的是R4(对应文件夹p)  R2正常(对应文件夹P2)
    train_O.train()