import queue
import threading

import cv2
import numpy as np

from bag_detection.utils import CustomStream, HEIGHT, WIDTH


outLineOffset = 65   #65 Box out of the fill area going to be sealed
exitLineOffset = 90  #90 Right most line the box leaving the camera view
boxOneLineOffset = 40 #40
beltLineOffset = 50  #50
scale_percent = 60  # percent of original size


class Camera:
    box = None
    outLine = None
    exitLine = None
    boxOneLine = None
    beltLine = None
    previousBag = None
    shift = False
    outAreaHadBoxBefore = False
    outAreaBoxBefore = None

    def __init__(self, cam_id, detector, box=None, main_cam=False,
                 show_window=True,
                 record=True,
                 width=WIDTH, height=HEIGHT):
        self.cam_id = cam_id
        self.detector = detector
        self.main_cam = main_cam
        self.reset(box)
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
            self.window = 'Camera_{}'.format("main" if self.main_cam else "second")
            cv2.namedWindow(self.window)

        if self.record:
            filename = "cam_{}.avi".format("main" if self.main_cam else "second")
            print("init writer: {}".format(filename))
            self.recorder = cv2.VideoWriter(filename,
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

        self.beltLine = self.drawXLine(beltLineOffset)

        if self.main_cam:
            self.outLine = self.drawYLine(outLineOffset)
            self.exitLine = self.drawYLine(exitLineOffset, (0, 0, 255))

        for label, bbox, score in zip(*predictions):
            if score < score_filter:
                continue

            if colors is not None:
                self.color = colors[label]

            detectedBag = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            cv2.rectangle(self.frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.color, thickness=2)

            if label == 'chyf_bag_2':
                if self.detectedBagCrossedLine(detectedBag):
                    self.box.bags += 1
                    print("Added new bag. Total {}".format(self.box.bags))

                self.registerPreviousBag(detectedBag)

            if label == 'out_box' and self.main_cam:
                self.update_box_v2(detectedBag)

            self.setHudText('{}: {}'.format(label, round(float(score), 2)), (bbox[0], bbox[1] - 10))

        k = cv2.waitKey(1)

        if k == 27:
            return False, self.shift
        return True, self.shift

    def detectedBagCrossedLine(self, detectedBag):
        """
        Check if detected bag has crossed the belt line
        Condition satisfied if bag was detected before and its coordinates are registered,
        coordinate Y of previous detection is less than belt line (above)
        and current bag coordinate Y is more than belt line (below)
        :param detectedBag:
        :return:
        """
        return self.previousBag is not None and self.previousBag[1] < self.beltLine < detectedBag[1]

    def registerPreviousBag(self, detectedBag):
        """
        Make registrations of bag detection when Y coordinate is before belt line minus 50 pixels
        or when it already crossed belt line.
        It will create a "blind" spot of 50 px to make sure we don't register same bag
        if it shakes and crosses a line multiple times
        :param detectedBag:
        :return:
        """
        if self.previousBag is None or (
                detectedBag[1] < self.beltLine - 50 or detectedBag[1] > self.beltLine):
            self.previousBag = detectedBag

    def reset(self, box=None):
        self.box = box
        if self.main_cam:
            self.box.box_name = "main"
        self.shift = False

    def stop(self):
        self.cap.stop()
        self.recorder.release()
        cv2.destroyAllWindows()

    def setHudText(self, text, org):
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
