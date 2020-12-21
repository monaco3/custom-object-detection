import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('opencv-python')
install('detecto')
install('imutils')

import os.path as op
from detecto import core

from bag_detection.belt import *

colors = {
    'chyf_bag_1': (255, 0, 0),
    'chyf_bag_2': (0, 0, 255),
    'chyf_bag_3': (0, 255, 0),
    'next_box': (255, 0, 0),
    'main_box': (0, 0, 255),
    'out_box': (0, 255, 0),
}

class_names = list(colors.keys())

model_name = 'chyf_bag_6cl'
model_number = 9
model_dir = 'detecto_models'
model_path = op.join(model_dir, model_name, 'model_{}.pth'.format(model_number))

dataset_path = 'data'

video_1 = "cam_1.mp4"
video_2 = "cam_2.mp4"
video_file_1 = op.join(dataset_path, video_1)
video_file_2 = op.join(dataset_path, video_2)


def main():
    detector = core.Model.load(model_path, class_names)

    belt = Belt(detector, video_file_1, video_file_2)
    belt.run(colors, threshold=0.35, target=25, show_window=False, record=True)


if __name__ == '__main__':
    main()
