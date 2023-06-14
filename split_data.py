import os  
import random   
  
sets = ['train', 'test', 'val']
xmlfilepath="G:/desktop/SAR_FZ1/labels_new/" # label文件的路径
saveBasePath="G:/desktop/SAR_FZ1/ImageSets/" # 生成的txt文件的保存路径
  
trainval_percent=0.9  # 训练验证集占整个数据集的比重（划分训练集和测试验证集）
train_percent=0.8     # 训练集占整个训练验证集的比重（划分训练集和验证集）
total_xml = os.listdir(xmlfilepath)  
num=len(total_xml)    
list=range(num)    
tv=int(num*trainval_percent)    
tr=int(tv*train_percent)    
trainval= random.sample(list,tv)    
train=random.sample(trainval,tr)    
  
print("train and val size",tv)   
ftrainval = open(os.path.join(saveBasePath,'Main/trainval.txt'), 'w')    
ftest = open(os.path.join(saveBasePath,'Main/test.txt'), 'w')    
ftrain = open(os.path.join(saveBasePath,'Main/train.txt'), 'w')    
fval = open(os.path.join(saveBasePath,'Main/val.txt'), 'w')    
  
for i  in list:    
    name=total_xml[i][:-4]+'\n'    
    if i in trainval:    
        ftrainval.write(name)    
        if i in train:    
            ftrain.write(name)    
        else:    
            fval.write(name)    
    else:    
        ftest.write(name)    
    

for image_set in sets:
    image_ids = open('G:/desktop/SAR_FZ1/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('G:/desktop/SAR_FZ1/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write('JPEGImages/%s.jpg\n' % (image_id))
    list_file.close()

ftrainval.close()    
ftrain.close()    
fval.close()    
ftest .close() 
print("Done!")
