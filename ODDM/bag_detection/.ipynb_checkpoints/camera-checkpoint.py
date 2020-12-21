import queue
import threading

import cv2
import numpy as np

from bag_detection.utils import CustomStream

WIDTH = 600
HEIGHT = 200

WHITE = (255, 255, 255)

outLineOffset = 65
exitLineOffset = 90
boxOneLineOffset = 40
beltLineOffset = 30
scale_percent = 60  # percent of original size


class Camera:
    main_box = None
    outLine = None
    exitLine = None
    boxOneLine = None
    beltLine = None
    targetBagBefore = None
    shift = False
    outAreaHadBoxBefore = False
    outAreaBoxBefore = None

    def __init__(self, cam_id, detector, box=None, tagsNumbers=0, main_cam=False,
                 show_window=True,
                 record=True,
                 width=WIDTH, height=HEIGHT):
        self.cam_id = cam_id
        self.detector = detector
        self.reset(box)
        self.tagsNumbers = tagsNumbers
        self.main_cam = main_cam
        self.frame = None
        self.color = (255, 0, 0)
        self.success = False
        self.overlay = None
        self.show_window = show_window
        self.record = record
        self.width = width
        self.height = height
        self.q = queue.Queue()

        try:
            self.cap = CustomStream(src=int(self.cam_id)).start()
        except ValueError:
            self.cap = CustomStream(src=self.cam_id).start()

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 8)

        if self.show_window:
            self.window = 'Camera' + str(self.cam_id)
            cv2.namedWindow(self.window)

        if self.record:
            print("init writer")
            self.recorder = cv2.VideoWriter("cam_record_{}.avi".format(self.cam_id),
                                            cv2.VideoWriter_fourcc(*'DIVX'),
                                            30, (frame_width, frame_height))

    def getCamera(self):
        return self.cap

    def run(self, score_filter=0.2, colors=None):

        self.success, frame = self.cap.read()
        if not self.success:
            return False, self.shift

        def frame_render(queue_from_cam, frame):
            queue_from_cam.put(frame)

        cam = threading.Thread(target=frame_render, args=(self.q, frame))
        cam.start()
        cam.join()
        self.frame = self.q.get()
        self.q.task_done()

        predictions = self.detector.predict(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        # print('Camera ' + str(self.cam_id) + ' Tag counter ' + str(self.tagsNumbers.size()))
        # print()

        self.beltLine = self.drawXLine(beltLineOffset)

        if self.main_cam:
            self.outLine = self.drawYLine(outLineOffset)
            self.exitLine = self.drawYLine(exitLineOffset, (0, 0, 255))

        for label, bbox, score in zip(*predictions):
            if score < score_filter:
                continue

            if colors is not None:
                self.color = colors[label]

            currentTarget = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            cv2.rectangle(self.frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.color, thickness=2)

            if label == 'chyf_bag_2':
                if self.targetBagBefore is not None and self.targetBagBefore[1] < self.beltLine < currentTarget[1]:
                    self.main_box.bags += 1
                    print("Added new bag. Total {}".format(self.main_box.bags))
                if self.targetBagBefore is None or (
                        currentTarget[1] < self.beltLine - 50 or currentTarget[1] > self.beltLine):
                    self.targetBagBefore = currentTarget

            if label == 'out_box' and self.main_cam:
                # self.update_box_v1(currentTarget)
                self.update_box_v2(currentTarget)

            self.putText('{}: {}'.format(label, round(score.item(), 2)), (bbox[0], bbox[1] - 10))

        k = cv2.waitKey(1)

        if k == 27:
            return False, self.shift
        return True, self.shift

    def reset(self, main_box=None):
        self.main_box = main_box
        self.main_box.box_name = "main"
        self.shift = False

    def stop(self):
        self.cap.stop()
        self.recorder.release()
        cv2.destroyAllWindows()

    def putText(self, text, org):
        cv2.putText(self.frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2)

    def render(self):
        frame = self.resizeFrame(scale_percent)
        if self.success and self.show_window:
            cv2.imshow(self.window, frame)
        if self.record:
            self.recorder.write(self.frame)

    def drawYLine(self, offset_from_middle_per_cent=50, color=(255, 0, 0)):
        height = np.size(self.frame, 0)
        width = np.size(self.frame, 1)

        offset = (width * offset_from_middle_per_cent) / 100
        lineX = int(offset)
        cv2.line(self.frame, (lineX, 0), (lineX, height), color, 2)
        cv2.line(self.frame, (lineX - 50, 0), (lineX - 50, height), (255, 150, 0), 2)
        return lineX

    def drawXLine(self, offset_from_middle_per_cent=50, color=(255, 0, 0)):
        height = np.size(self.frame, 0)
        width = np.size(self.frame, 1)

        offset = (height * offset_from_middle_per_cent) / 100
        lineY = int(offset)
        cv2.line(self.frame, (0, lineY), (width, lineY), color, 2)
        cv2.line(self.frame, (0, lineY - 50), (width, lineY - 50), (255, 0, 150), 2)
        return lineY

    def resizeFrame(self, scale):
        width = int(self.frame.shape[1] * scale / 100)
        height = int(self.frame.shape[0] * scale / 100)
        dim = (width, height)
        return cv2.resize(self.frame, dim, interpolation=cv2.INTER_AREA)

    def update_box_v1(self, target):
        outAreaHasBox = False
        if self.outLine < target[0] < self.exitLine:
            outAreaHasBox = True

        if not outAreaHasBox and self.outAreaHadBoxBefore and self.exitLine - 50 < target[0]:
            print("Filled box moved out to next stage")
            self.outAreaHadBoxBefore = False

        elif outAreaHasBox and not self.outAreaHadBoxBefore:
            print("Box is filled and moved to the out area")
            self.outAreaHadBoxBefore = True
            self.shift = True

    def update_box_v2(self, target):
        if self.outAreaBoxBefore is None:
            if self.outLine < target[0] < self.exitLine - 50:
                self.outAreaBoxBefore = target[0]
                self.shift = True
                print("Box is filled and moved to the out area")

        elif self.outLine < self.outAreaBoxBefore < self.exitLine < target[0]:
            self.outAreaBoxBefore = None
            print("Filled box moved out to next stage")
