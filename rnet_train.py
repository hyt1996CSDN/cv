import train_box1
import net
if __name__ == '__main__':
    r_net=net.Net3_R()
    train_r=train_box1.trainer(r_net,r'C:/celeba3/48','./r_five2.pt',24,0.5,0.5)
    train_r.train()

