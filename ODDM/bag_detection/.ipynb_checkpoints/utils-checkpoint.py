import cv2
from imutils.video import WebcamVideoStream, FileVideoStream

class Box:
    def __init__(self, box_id, bags=0):
        self.box_id = box_id
        self.box_name = box_id
        self.bags = bags
        self.isAway = False
        self.isOut = False

    def getBags(self):
        return self.bags

    def bagsToString(self):
        return "Box {} has {} bags".format(self.box_name, self.bags)

    def printBags(self):
        print(self.bagsToString())


class Set:

    def __init__(self):
        self.set = set([])

    def add(self, element):
        self.set.add(element)

    def size(self):
        return len(self.set)

    def contains(self, element):
        return self.set.issuperset({element})


class CustomStream:
    def __init__(self, src=0, use_cv2=False):
        if use_cv2:
            self.obj = cv2.VideoCapture(src)
        elif src == 0:
            self.obj = WebcamVideoStream(src)
        elif src != 0:
            self.obj = FileVideoStream(src)

    def isOpened(self):
        if isinstance(self.obj, cv2.VideoCapture):
            return self.obj.isOpened()
        return self.obj.stream.isOpened()

    def start(self):
        self.obj.start()
        return self

    def update(self):
        self.obj.update()

    def read(self):
        if isinstance(self.obj, cv2.VideoCapture):
            return self.obj.read()
        return not self.obj.stopped, self.obj.read()

    def stop(self):
        self.obj.stop()
        if isinstance(self.obj, cv2.VideoCapture):
            self.obj.release()
        else:
            self.obj.stream.release()

    def set(self, propId, value):
        if isinstance(self.obj, cv2.VideoCapture):
            self.obj.set(propId, value)
            return
        self.obj.stream.set(propId, value)

    def get(self, propId):
        if isinstance(self.obj, cv2.VideoCapture):
            self.obj.get(propId)
            return
        return self.obj.stream.get(propId)