import numpy as np
d1=[1,1,4,4]
d2=[2,2,6,6]

def iou(a,b,c=0):
    x1=np.maximum(a[0],b[:,0])
    y1=np.maximum(a[1],b[:,1])
    x2=np.minimum(a[2],b[:,2])
    y2=np.minimum(a[3],b[:,3])
    w=np.maximum(x2-x1,0)
    h=np.maximum(y2-y1,0)
    s_1=(a[2]-a[0])*(a[3]-a[1])
    s_2=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    s_min=np.minimum(s_1,s_2)
    s_binji=w*h
    s_jiaoji=s_1+s_2-s_binji
    if c==0:
        value_ious=s_binji/s_jiaoji
    else:
        value_ious=s_binji/s_min
    return value_ious

def nsm(a,b,c=0):
    if a.shape[0]==0:
        return np.array([])
    box_1=a[(-a[:,4]).argsort()]
    box_finall=[]
    while box_1.shape[0]>0:
        box_2=box_1[0]
        box_3=box_1[1:]
        box_finall.append(box_1[0])
        index=np.where(iou(box_2,box_3,c)<b)
        box_1=box_3[index]
    return box_finall

def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side*1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side*1

    return square_bbox
