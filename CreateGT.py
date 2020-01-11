import argparse
import cv2
import numpy as np

class CreateGT:
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
        gt = np.zeros((end, 4), dtype=int)        
        while self.cap.isOpened():
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))            
            print(pos, gt[pos])

            ret, frame = self.cap.read()
            self.shape = frame.shape
            if ret:
                cv2.imshow(self.image_path, frame)
                cv2.setMouseCallback(self.image_path, CreateGT.mouse_event_handler, gt[pos])
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(pos + 1, end))
                elif key == ord('p'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - 1, 0))
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            else:
                break

        cv2.destroyAllWindows()
        self.save_gt(gt)

    def save_gt(self, gt):
        print(gt)
        gt[..., 0] = np.clip(gt[..., 0], 0, self.shape[1])
        gt[..., 1] = np.clip(gt[..., 1], 0, self.shape[0])
        gt[..., 2] = np.clip(gt[..., 2], 0, self.shape[1])
        gt[..., 3] = np.clip(gt[..., 3], 0, self.shape[0])
        
        gt[..., 2] = gt[..., 2] - gt[..., 0]
        gt[..., 3] = gt[..., 3] - gt[..., 1]

        np.savetxt(self.name, gt, delimiter=',', fmt='%d')

    @staticmethod
    def mouse_event_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param[0] = x
            param[1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            param[2] = x
            param[3] = y
            if param[0] > x:
                param[2] = param[0]
                param[0] = x
            if param[1] > y:
                param[3] = param[1]
                param[1] = y
            print(param)

def main(argvs):
    cgt = CreateGT()
    cgt.run()

if __name__=='__main__':
    main('')
