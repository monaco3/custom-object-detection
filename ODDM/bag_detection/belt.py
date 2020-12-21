from __future__ import division
from __future__ import print_function

from PIL import ImageFile

from bag_detection.camera import Camera
from bag_detection.utils import Box

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Belt:

    def __init__(self, model, cam_src_1=2, cam_src_2=0):
        self.model = model
        self.cam_src_1 = cam_src_1
        self.cam_src_2 = cam_src_2

    def run(self, colors, threshold, target, *args, **kwargs):

        box_counter = 0
        main_box = Box(box_counter)

        box1 = Box(box_counter + 1)
        box2 = Box(box_counter + 2)
        box3 = Box(box_counter + 3)
        box4 = Box(box_counter + 4)

        cam1 = Camera(self.cam_src_1, self.model, box=main_box, main_cam=True, *args, **kwargs)
        cam2 = Camera(self.cam_src_2, self.model, box=box4, *args, **kwargs)

        while True:
            cam1running, shift = cam1.run(colors=colors, score_filter=threshold)
            cam2running, _ = cam2.run(colors=colors, score_filter=threshold)

            if not cam1running:
                break

            # box1.printBags()
            # box2.printBags()
            # box3.printBags()
            # box4.printBags()

            cam1.setHudText(main_box.bagsToString(), (10, 50))
            cam1.setHudText(box1.bagsToString(), (10, 65))
            cam1.setHudText(box2.bagsToString(), (10, 80))
            cam1.setHudText(box3.bagsToString(), (10, 95))
            cam1.setHudText(box4.bagsToString(), (10, 110))

            cam1.render()
            cam2.render()

            if cam1.box.isAway and cam1.box.getBags() != target:
                print("Error recorded - BOX DOES NOT HAVE ENOUGH BAGS")

            main_box.printBags()

            if shift:
                # box1.printBags()
                box_counter += 1
                main_box = Box(box1.box_id, box1.getBags())

                box1 = Box(box2.box_id, box2.getBags())
                box2 = Box(box3.box_id, box3.getBags())
                box3 = Box(box4.box_id, box4.getBags())
                box4 = Box(box_counter + 3)
                cam1.reset(main_box)
                cam2.reset(box4)

        print("Exit")

        cam1.stop()
        cam2.stop()
