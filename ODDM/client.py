import subprocess
import sys

sys.path.append('../YoloV3')
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install('opencv-python')
# install('detecto')
# install('imutils')
# install('tqdm')

import os.path as op
import argparse
from datetime import datetime
from datetime import timedelta


# from detecto import core
# import read_stream
# from read_stream import*
from bag_detection.belt import *
from bag_detection.utils import HEIGHT, WIDTH
from yolov3 import YoloV3, getOpts
from read_stream import my_streams


#Assign kinesis video stream names. Note either use StreamName or ARNname but not both
stream_name_1 = "4ed549044d107c351edc8e2c70429bbf"
stream_name_2 = "856367bcd07648b198f7295de9e42d38"
arn_1 = "arn:aws:kinesisvideo:eu-west-1:435342033141:stream/4ed549044d107c351edc8e2c70429bbf/1595242384068"
arn_2 = "arn:aws:kinesisvideo:eu-west-1:435342033141:stream/856367bcd07648b198f7295de9e42d38/1595242418735"


#Colours of the bags/detected pouches and their positions
colors = {
    'chyf_bag_1': (255, 0, 0),  #Bag on the belt
    'chyf_bag_2': (0, 0, 255),  #Bag on the robot arm
    'chyf_bag_3': (0, 255, 0),  #Bag in the Box
    'next_box': (255, 0, 0),    #The nex bag to be filled (in the main camera)
    'main_box': (0, 0, 255),    #The main box being filled up
    'out_box': (0, 255, 0),     #The filled box the is leaving the cartoning area
}

class_names = list(colors.keys())

model_name = 'chyf_bag_6cl'
model_number = 9
model_dir = 'detecto_models'
model_path = op.join(model_dir, model_name, 'model_{}.pth'.format(model_number))

dataset_path = 'data'

#Using cameras connected to the machine
# cam_1_src = 0
# cam_2_src = 1 

#Using Saved video file
# video_1 = "cam_1.mp4"
# video_2 = "cam_2.mp4"
# cam_1_src = op.join(dataset_path, video_1)
# cam_2_src = op.join(dataset_path, video_2)

#Using kinesis video stream
cam_1_src = my_streams("main_camera", stream_name_1)
cam_2_src = my_streams("second_camera", stream_name_2)



#Change the yolo weights and configurations for use. Remember the correct paths

def main():
    filename = "yolov3-chyfbags_4cl-tiny"
    opt = getOpts()
    print(opt)
    opt.cfg = op.join(opt.root, "cfg/{}.cfg".format(filename))
    opt.names = op.join(opt.root, "data/chyfbags_4cl.names")
    opt.weights = op.join(opt.root, "weights/{}_final.weights".format(filename))
    opt.source = "data\chyf_bag_test.mp4"
    opt.view_img = True
    opt.resize = 50

    # detector = core.Model.load(model_path, class_names)
    detector = YoloV3(opt)

    belt = Belt(detector, cam_1_src, cam_2_src)
    belt.run(colors, threshold=0.25, target=25, show_window=True, record=True)


if __name__ == '__main__':
    main()
