import os, shutil, subprocess, json
import random
import config as cfg

# label_map = ['car', 'truck', 'bus', 'pedestrian', 'cyclist']
label_map = cfg.labels

with open(cfg.raw_ann) as fp:
    d1 = json.load(fp)

shutil.rmtree(cfg.data_path, ignore_errors=True)

train_img = cfg.data_path+'/images/train'
train_ann = cfg.data_path+'/labels/train'
val_img = cfg.data_path+'/images/val'
val_ann = cfg.data_path+'/labels/val'

for dirs in [train_ann, train_img, val_ann, val_img]:
    os.makedirs(dirs, exist_ok=True)

img_w, img_h = d1['meta_info']['image_size']
for frame in d1['frames']:
    if 'annos' not in frame:
        continue

    labels = frame['annos']['names']
    file_name = frame['frame_id']+'.txt'
    file_name = os.path.join(train_ann, file_name)
    write_lines = []
    for x, bbox in enumerate(frame['annos']['boxes_2d'][cfg.cam]):
        if sum(bbox) > 0:
            label = labels[x]
            if label.lower() in label_map:
                idx = int(label_map.index(label.lower()))
                x_c = ((bbox[0]+bbox[2])/2)/img_w
                y_c = ((bbox[1]+bbox[3])/2)/img_h
                w = (bbox[2]-bbox[0])/img_w
                h = (bbox[3]-bbox[1])/img_h
                write_lines.append(' '.join([str(it) for it in [idx, x_c, y_c, w, h]]))
    if write_lines:
        with open(file_name, 'a+') as fp:
            fp.writelines('\n'.join(write_lines))

# Copy images

l1 = [it.replace('.txt', '.jpg') for it in os.listdir(train_ann)]

for i in l1:
    src1 = os.path.join(cfg.raw_images, i)
    dst1 = os.path.join(train_img, i)
    shutil.copy(src1, dst1)
# Splitting train and val

l1 = [os.path.join(train_ann, it) for it in os.listdir(train_ann)]

random.shuffle(l1)

l1 = l1[:int(0.2*len(l1))]

for i in l1:
    s1 = i
    s2 = i.replace('/labels', '/images')
    d1 = s1.replace('/train', '/val')
    d2 = s2.replace('/train', '/val')
    os.rename(s1, d1)
    os.rename(s2.replace('.txt', '.jpg'), d2.replace('.txt', '.jpg'))

try:
    os.chdir('./yolov5')

except:
    print('Error')

subprocess.call(["python3", "train.py", "--batch", "-1", "--img", "1280", "--epochs", "100", "--data", "once.yaml", "--weights", "''", "--cfg", "yolov5s.yaml"])