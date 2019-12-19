import torch.nn as nn

class Net_p(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_p1=nn.Sequential(
            nn.Conv2d(3,10,3),
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(3,2,padding=1),
            nn.PReLU(), #5*5
            nn.Conv2d(10,16,3),
            nn.BatchNorm2d(16),
            nn.PReLU(), #3*3
            nn.Conv2d(16,32,3),
            nn.BatchNorm2d(32),
            nn.PReLU()  #1*1
        )
        self.f_p2 = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.f_p3 = nn.Sequential(
            nn.Conv2d(32, 4, 1)
        )

    def forward(self,x):
        y=self.f_p1(x)
        y1=self.f_p2(y)
        y2=self.f_p3(y)
        return y1,y2

class Net_R(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_r1=nn.Sequential(
            nn.Conv2d(3,28,3),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(3,2,padding=1),
            nn.Conv2d(28,48,3),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(48,64,2),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.f_r2 = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
            nn.BatchNorm1d(1), #p5没这
            nn.Sigmoid()
        )

        self.f_r3 = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, input):
        y=self.f_r1(input)
        y=y.reshape(-1,3*3*64)
        y1=self.f_r2(y)
        y2=self.f_r3(y)
        return y1,y2

class Net_O(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_o1=nn.Sequential(
            nn.Conv2d(3,32,3), #48
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3,2,padding=1),
            nn.Conv2d(32,64,3), #23
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(64,64,3),  #10
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,2), #4
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        self.f_o2=nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),  # 3
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.f_o3 = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),  # 3
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 4)
        )

    def forward(self,input):
        y=self.f_o1(input)
        y=y.reshape(-1,3*3*128)
        y1=self.f_o2(y)
        y2=self.f_o3(y)
        return y1,y2
#-----------------------------------------------------------------

class Net2_p(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_p1=nn.Sequential(
            nn.Conv2d(3,10,3),
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(3,2,padding=1),
            nn.PReLU(), #5*5
            nn.Conv2d(10,16,3),
            nn.BatchNorm2d(16),
            nn.PReLU(), #3*3
            nn.Conv2d(16,32,3),
            nn.BatchNorm2d(32),
            nn.PReLU()  #1*1
        )
        self.f_p2 = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.f_p3 = nn.Sequential(
            nn.Conv2d(32, 14, 1)
        )

    def forward(self,x):
        y=self.f_p1(x)
        y1=self.f_p2(y)
        y2=self.f_p3(y)
        return y1,y2

class Net2_R(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_r1=nn.Sequential(
            nn.Conv2d(3,28,3),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(3,2,padding=1),
            nn.Conv2d(28,48,3),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(48,64,2),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.f_r2 = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
            nn.BatchNorm1d(1), #p5没这
            nn.Sigmoid()
        )

        self.f_r3 = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU(),
            nn.Linear(128, 14)
        )

    def forward(self, input):
        y=self.f_r1(input)
        y=y.reshape(-1,3*3*64)
        y1=self.f_r2(y)
        y2=self.f_r3(y)
        return y1,y2

class Net2_O(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_o1=nn.Sequential(
            nn.Conv2d(3,32,3), #48
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3,2,padding=1),
            nn.Conv2d(32,64,3), #23
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(64,64,3),  #10
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,2), #4
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        self.f_o2=nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),  # 3
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256,1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.f_o3 = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),  # 3
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 14)
        )

    def forward(self,input):
        y=self.f_o1(input)
        y=y.reshape(-1,3*3*128)
        y1=self.f_o2(y)
        y2=self.f_o3(y)
        return y1,y2
#---------------------------------------------------------------------
# class Net3_p(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.f_p1=nn.Sequential(
#             nn.Conv2d(3,10,3,3),
#             nn.BatchNorm2d(10),
#             nn.PReLU(),
#             nn.Conv2d(10,16,2,2),
#             nn.BatchNorm2d(16),
#             nn.PReLU(), #3*3
#             nn.Conv2d(16,32,2,2),
#             nn.BatchNorm2d(32),
#             nn.PReLU()  #1*1
#         )
#         self.f_p2 = nn.Sequential(
#             nn.Conv2d(32, 1, 1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.f_p3 = nn.Sequential(
#             nn.Conv2d(32, 4, 1)
#         )
#
#     def forward(self,x):
#         y=self.f_p1(x)
#         y1=self.f_p2(y)
#         y2=self.f_p3(y)
#         return y1,y2
#
# class Net3_R(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.f_r1=nn.Sequential(
#             nn.Conv2d(3,28,3),
#             nn.BatchNorm2d(28),
#             nn.PReLU(),
#             nn.MaxPool2d(3,2,padding=1),
#             nn.Conv2d(28,48,3),
#             nn.BatchNorm2d(48),
#             nn.PReLU(),
#             nn.MaxPool2d(3,2),
#             nn.Conv2d(48,64,2),
#             nn.BatchNorm2d(64),
#             nn.PReLU()
#         )
#
#         self.f_r2 = nn.Sequential(
#             nn.Linear(3 * 3 * 64, 128),
#             nn.PReLU(),
#             nn.Linear(128, 1),
#             nn.BatchNorm1d(1), #p5没这
#             nn.Sigmoid()
#         )
#
#         self.f_r3 = nn.Sequential(
#             nn.Linear(3 * 3 * 64, 128),
#             nn.PReLU(),
#             nn.Linear(128, 14)
#         )
#
#     def forward(self, input):
#         y=self.f_r1(input)
#         y=y.reshape(-1,3*3*64)
#         y1=self.f_r2(y)
#         y2=self.f_r3(y)
#         return y1,y2
#
# class Net3_O(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.f_o1=nn.Sequential(
#             nn.Conv2d(3,32,3), #48
#             nn.BatchNorm2d(32),
#             nn.PReLU(),
#             nn.MaxPool2d(3,2,padding=1),
#             nn.Conv2d(32,64,3), #23
#             nn.BatchNorm2d(64),
#             nn.PReLU(),
#             nn.MaxPool2d(3,2),
#             nn.Conv2d(64,64,3),  #10
#             nn.BatchNorm2d(64),
#             nn.PReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64,128,2), #4
#             nn.BatchNorm2d(128),
#             nn.PReLU()
#         )
#
#         self.f_o2=nn.Sequential(
#             nn.Linear(3 * 3 * 128, 256),  # 3
#             nn.BatchNorm1d(256),
#             nn.PReLU(),
#             nn.Linear(256,1),
#             nn.BatchNorm1d(1),
#             nn.Sigmoid()
#         )
#
#         self.f_o3 = nn.Sequential(
#             nn.Linear(3 * 3 * 128, 256),  # 3
#             nn.BatchNorm1d(256),
#             nn.PReLU(),
#             nn.Linear(256, 14)
#         )
#
#     def forward(self,input):
#         y=self.f_o1(input)
#         y=y.reshape(-1,3*3*128)
#         y1=self.f_o2(y)
#         y2=self.f_o3(y)
#         return y1,y2