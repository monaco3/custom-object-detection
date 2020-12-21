# https://towardsdatascience.com/build-a-custom-trained-object-detection-model-with-5-lines-of-code-713ba7f6c0fb
import os

import torch
import os.path as op
from PIL import ImageFile
from detecto import core, visualize, utils

print("Cuda is available" if torch.cuda.is_available() else "Cuda is not available")

ImageFile.LOAD_TRUNCATED_IMAGES = True
dataset_path = 'E:/ProgrammingProjects/PycharmProjects/dataset'
result_dir = 'detecto_models'

model_ = None


# kangaroo_train = [dataset_path + '/kangaroo/train/annots', dataset_path + '/kangaroo/train/images']
# kangaroo_val = [dataset_path + '/kangaroo/validate/annots', dataset_path + '/kangaroo/validate/images']
# kangaroo_test = [dataset_path + '/kangaroo/test/annots', dataset_path + '/kangaroo/test/images']


def train(model_name, class_names, load_model_nr, epochs, callback=None):
    train_dataset = core.Dataset(op.join(dataset_path, model_name, 'train'))
    val_dataset = core.Dataset(op.join(dataset_path, model_name, 'validate'))
    save_dir = op.join(result_dir, model_name)

    print("Training set: {}".format(len(train_dataset)))
    print("Validation set: {}".format(len(val_dataset)))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = core.Model.load(op.join(save_dir, 'model_{}.pth'.format(load_model_nr)), class_names)
    model.fit(train_dataset, val_dataset, save_path=save_dir, start_epoch=load_model_nr, epochs=load_model_nr + epochs,
              verbose=True, callback=callback)
    return model


def run_cam(model_path, class_names):
    model = core.Model.load(model_path, class_names)
    visualize.detect_live(model)


def run_video(model, model_name, video_file, output_file, score_filter=0.4):
    video_file = op.join(dataset_path, model_name, video_file)
    output_file = op.join(dataset_path, model_name, output_file)
    if os.path.exists(video_file):
        colors = {
            'chyf_bag_1': (255, 0, 0),
            'chyf_bag_2': (0, 0, 255),
            'chyf_bag_3': (0, 255, 0),
            'next_box': (255, 0, 0),
            'main_box': (0, 0, 255),
            'out_box': (0, 255, 0)
        }
        visualize.detect_video(model, video_file, output_file, score_filter=score_filter, colors=colors,
                               callback=run_img)


def run_img(model, img_path=None, image=None, score_filter=0.4):
    if img_path is None:
        print("Call run_img function")
        img_path = op.join(dataset_path, 'chyf_bag_3cl/frames/frame177.jpg')

    if not os.path.exists(img_path) and image is None:
        print("No image and path is not valid {}".format(img_path))
        return

    if image is None:
        image = utils.read_image(img_path)

    colors = {
        'chyf_bag_1': 'b',
        'chyf_bag_2': 'r',
        'chyf_bag_3': 'g',
        'next_box': 'b',
        'main_box': 'r',
        'out_box': 'g'
    }
    visualize.predict_image(model, image, fontsize=5, colors=colors, score_filter=score_filter)


if __name__ == '__main__':
    class_names = ['chyf_bag_1', 'chyf_bag_2', 'chyf_bag_3', 'next_box', 'main_box', 'out_box']
    model_name = 'chyf_bag_3cl'
    model_number = 9
    threshold = 0.4
    model_path = op.join(result_dir, model_name, 'model_{}.pth'.format(model_number))

    # model_ = train(model_name=model_name,
    #                class_names=class_names,
    #                load_model_nr=model_number,
    #                epochs=50,
    #                callback=run_img)

    if model_ is None:
        model_ = core.Model.load(model_path, class_names)

    run_video(model_, model_name, model_name + '_test.mp4', 'detected_ch_yoflex.avi', score_filter=threshold)
    # run_img(model_, score_filter=threshold)
    # utils.split_video(op.join(dataset_path, model_name, 'training_video.mp4'),
    #                   op.join(dataset_path, model_name, 'frames'), step_size=2)
