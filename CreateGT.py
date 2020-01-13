import argparse
import cv2
import numpy as np

class CreateGT(object):
    class CGT:
        gt = None
        pos = 0
        img_size = (0, 0)
        img_name = None
        frame = None
        
    def __init__(self):
        opt = self.parse_config()

        self.image_path = opt.image_path        
        
        if self.image_path.lower().endswith(('.png', '.jpg')):
            self.name = self.image_path.replace(".png", ".txt").replace(".jpg", ".txt")
        elif self.image_path.lower().endswith(('.avi', '.mp4')):
            self.name = 'groundtruth_rect.txt'
        else:
            assert False, 'Error : Unknown file format!!!'
        
        self.load_image(self.image_path)
        
    def parse_config(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--image_path", default="dog.jpg", help="Path to the image file")
        args, _ = ap.parse_known_args()
        return args

    def load_image(self, path):
        self.cap = cv2.VideoCapture(path) 
        assert self.cap.isOpened(), 'Cannot open source'
        
    def run(self):
        end = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gt = self.CGT()
        gt.gt = np.zeros((end, 4), dtype=int)
        gt.img_name = self.image_path

        while self.cap.isOpened():
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))        
            ret, frame = self.cap.read()
            if ret:
                gt.pos = pos
                gt.frame = frame      
                print(pos, gt.gt[pos])

                self.shape = frame.shape
                gt.img_size = (frame.shape[1], frame.shape[0])

                cv2.imshow(self.image_path, frame)
                cv2.setMouseCallback(self.image_path, CreateGT.mouse_event_handler, gt)
                key = cv2.waitKey() & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos + 1, end - 1))
                elif key == ord('p'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - 1, 0))
                elif key != 0xFF:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            else:
                break

        cv2.destroyAllWindows()
        self.save_gt(gt.gt)

    def save_gt(self, gt):        
        np.savetxt(self.name, gt, delimiter=',', fmt='%d')
        print(gt)

    @staticmethod
    def mouse_event_handler(event, x, y, flags, param):
        gt = param
        rect = gt.gt[gt.pos]
        x = max(0, min(x, gt.img_size[0] - 1))
        y = max(0, min(y, gt.img_size[1] - 1))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            rect[0] = x
            rect[1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            rect[2] = abs(rect[0] - x)
            rect[3] = abs(rect[1] - y)
            if rect[0] > x:
                rect[0] = x
            if rect[1] > y:                
                rect[1] = y
            
            img = gt.frame.copy()
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2]+rect[0], rect[3]+rect[1]), (0, 255, 0), 1)
            cv2.imshow(gt.img_name, img)
            cv2.displayStatusBar(gt.img_name, f'{gt.pos} : {rect}')
            print(gt.pos, rect)

def main(argvs):
    cgt = CreateGT()
    cgt.run()

if __name__=='__main__':
    main('')
