split = 'train'
batch = '000104'

data_path = './data'

yolo_repo = './yolov5'

cam = 'cam01'

labels = ['car'] # specify list of labels to be considered for training

raw_images = './raw_images/'+split+'/'+batch+'/'+cam
raw_ann = './raw_annotations/'+split+'/'+batch+'.json'
