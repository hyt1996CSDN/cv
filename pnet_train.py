import train_box1
import train_box2
import net
if __name__ == '__main__':
    p_net=net.Net2_p()
    train_p=train_box2.trainer(p_net,r'C:/celeba4/48','./p__gai1.pt',12)#p只有2没4
    train_p.train()