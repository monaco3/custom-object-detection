import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import paddedResize
from utils.utils import *


def getOpts():
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=root, help='root dir')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples',
                        help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--resize', action='store_true', help='resize window', default=100)
    return parser.parse_args()

class YoloV3:

    def __init__(self, opt=None):
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        self.model = Darknet(opt.cfg, opt.img_size)
        self.opt = opt
        source = opt.source
        self.webcam = source == '0' \
                      or source.startswith('rtsp') \
                      or source.startswith('http') \
                      or source.endswith('.txt') \
                      or source.endswith('/stdin')

        attempt_download(opt.weights)

        if opt.weights.endswith('.pt'):
            self.model.load_state_dict(torch.load(opt.weights, map_location=self.device)['model'])
        else:
            _ = load_darknet_weights(self.model, opt.weights)

        self.model.to(self.device).eval()

        half = opt.half and self.device.type != 'cpu'
        if half:
            self.model.half()

        self.names = load_classes(opt.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def predict(self, img0):
        img = paddedResize(img0, self.opt.img_size, self.opt.half)
        pred, img_t = self.getPredictions(img)
        return self.getDetections(pred, img_t, img0)

    def getPredictions(self, img):
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]

        if self.opt.half:
            pred = pred.float()

        return non_max_suppression(pred, self.opt.conf_thres, self.opt.nms_thres), img

    def getDetections(self, pred, img, im0s):
        predictions = []

        for i, det in enumerate(pred):
            if self.webcam:  # batch_size >= 1
                index, im0 = '%g: ' % i, im0s[i]
            else:
                index, im0 = '', im0s

            if det is not None and len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                labels = []
                scores = []
                bboxes = []

                for *bbox, score, cls in det:
                    labels.append(self.names[int(cls)])
                    scores.append(float(score))
                    bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

                predictions.append((labels, bboxes, scores))
        try:
            return predictions[0]
        except Exception:
            return []
