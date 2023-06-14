
import os
import cv2
import numpy as np
from tqdm import tqdm
import colorsys
def xywhn2xyxy(x, w=512, h=512, padw=0, padh=0):
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

roles = [
        "HM", 
        "HL", 
        "YL", 
        "QT"] 
hsv_tuples = [
    (x / len(roles), 1., 1.)
        for x in range(len(roles))
]  # 获得hsv格式的不同色度

colors = list(
    map(
        lambda x: colorsys.hsv_to_rgb(*x),
        hsv_tuples
    )
)  # 获得rgb格式的不同颜色
colors = list(
    map(
        lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors
    )
)

imgfilefolder = r"G:\desktop\SAR_FZ1\JPEGImages"
txtfilefolder = r"G:\desktop\SAR_FZ1\labels_new"
savefilefoler = r"G:\desktop\SAR_FZ1\val"
if not os.path.exists(savefilefoler):
    os.makedirs(savefilefoler)
imgs = os.listdir(txtfilefolder)
for i in tqdm(imgs):
    if i.endswith('.txt'):
        flags = [0, 0, 0, 0]
        name = i.split('.')[0]
        imgPath = os.path.join(imgfilefolder, name+'.jpg')
        txtPath = os.path.join(txtfilefolder, name+'.txt')

        image = cv2.imread(imgPath)

        with open(txtPath, 'r') as f:
           # lines = f.readlines()
           # print(lines)
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        if lb.size>0:
            h, w = image.shape[:2]
            lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  # 反归一化

        for _, x in enumerate(lb):
            x = [int(i) for i in x]
            color_ = colors[int(x[0])]
            cv2.rectangle(image,(x[1], x[2]), (x[3], x[4]), color_, thickness=3)
            cv2.putText(image, roles[int(x[0])],(x[1], x[2]),cv2.FONT_HERSHEY_COMPLEX, 1, color=color_, thickness=3)
        savePath = savefilefoler + '\\' + name + '.jpg'
        cv2.imwrite(savePath, image)