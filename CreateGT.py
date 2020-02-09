import argparse
import cv2
import numpy as np
import os
import glob

class CreateGT(object):
    class CGT:
        gt = None
        pos = 0
        img_size = (0, 0)
        img_name = None
        frame = None
        mode = None

    def parse_config(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--mode", default="cgt_yot", help="vtoi:video to image files, vyolo:view yolo gt, \
                        vrolo: view rolo rt, cgt_yolo:create gt for yolo, cgt_yot:create gt for yot")
        ap.add_argument("--image_path", default="../Tennis/Tennis_01/images", help="Path to the image/video file or dir")
        args, _ = ap.parse_known_args()
        return args
    
    def __init__(self):
        opt = self.parse_config()

        self.image_path = opt.image_path
        self.mode = opt.mode  
            
    def check_video_file(self, path):
        if not path.lower().endswith(('.avi', '.mp4')):
            assert False, 'Error : Unknown file format!!!'

    def vtoi(self, path):
        self.check_video_file(path)

        img_path = os.path.join(os.path.dirname(path), 'images')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        
        self.cap = cv2.VideoCapture(self.image_path) 
        assert self.cap.isOpened(), 'Cannot open source'
        while self.cap.isOpened():
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))    
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(os.path.join(img_path, '{:04d}.jpg'.format(pos+1)), frame)
                print(f'saved {os.path.join(img_path, "{:04d}.jpg".format(pos+1))}')
            else:
                break;        
    
    def draw_yolo_rect(self, img, path):
        label_path = path.replace(".png", ".txt").replace(".jpg", ".txt")
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            for l in labels:
                x1 = int(l[1]*img.shape[1] - l[3]*img.shape[1]/2)
                y1 = int(l[2]*img.shape[0] - l[4]*img.shape[0]/2)
                x2 = int(l[1]*img.shape[1] + l[3]*img.shape[1]/2)
                y2 = int(l[2]*img.shape[0] + l[4]*img.shape[0]/2)
                print(l)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow(path, img)
    
    def vyolo(self, path):
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))]
        pos = 0
        end = len(files)
        isRun = True
        while(isRun):
            f = os.path.join(path, files[pos])
            cap = cv2.VideoCapture(f) 
            assert cap.isOpened(), 'Cannot open source'
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f, frame)
                    self.draw_yolo_rect(frame, f)
                    key = cv2.waitKey() & 0xFF

                    if key == ord('q'):
                        isRun = False
                        break
                    elif key == ord('n'):
                        pos = min(pos + 1, end - 1)
                    elif key == ord('p'):
                        pos = max(pos - 1, 0)                        
                else:
                    break
                    
            cv2.destroyAllWindows()
            print(f)

        pass

    def draw_rolo_rect(self, img, img_path, lbl_path):
        if os.path.exists(lbl_path):
            try:
                labels = np.loadtxt(lbl_path, delimiter='\t', dtype=float).reshape(-1, 4)
            except:
                labels = np.loadtxt(lbl_path, delimiter=',', dtype=float).reshape(-1, 4)
            for l in labels:
                x1 = int(l[0])
                y1 = int(l[1])
                x2 = int(l[0] + l[2])
                y2 = int(l[1] + l[3])
                print(l)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow(img_path, img)
    
    def vrolo(self, path):
        img_path = os.path.join(path, 'images')
        lbl_path = os.path.join(path, 'labels')
        img_files = sorted(glob.glob("%s/*.*" % img_path)) 
        lbl_files = sorted(glob.glob("%s/*.*" % lbl_path))
        
        pos = 0
        end = min(len(img_files), len(lbl_files))
        isRun = True
        while(isRun):
            img_f = img_files[pos]
            lbl_f = lbl_files[pos]
            cap = cv2.VideoCapture(img_f) 
            assert cap.isOpened(), 'Cannot open source'
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(img_f, frame)
                    self.draw_rolo_rect(frame, img_f, lbl_f)
                    key = cv2.waitKey() & 0xFF

                    if key == ord('q'):
                        isRun = False
                        break
                    elif key == ord('n'):
                        pos = min(pos + 1, end - 1)
                    elif key == ord('p'):
                        pos = max(pos - 1, 0)                        
                else:
                    break
                    
            cv2.destroyAllWindows()
            print(img_f, lbl_f)

        pass

    def cgt(self, path):
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg'))]
        files.sort()
        pos = 0
        end = len(files)
        isRun = True
        gt = self.CGT()
        if self.mode == 'cgt_yot':            
            name = os.path.join(os.path.dirname(path), 'groundtruth_rect.txt')
            if os.path.exists(name):
                gt.gt = np.loadtxt(name, delimiter=',')
            else:
                gt.gt = np.zeros((end, 4), dtype=float)
        gt.img_name = self.image_path

        while(isRun):
            f = os.path.join(path, files[pos])
            cap = cv2.VideoCapture(f) 
            assert cap.isOpened(), 'Cannot open source'        
            while cap.isOpened():
                ret, frame = cap.read()                
                if ret:
                    gt.pos = pos
                    gt.frame = frame
                    gt.mode = self.mode
                    print(pos, gt.gt[pos])

                    self.shape = frame.shape
                    gt.img_size = (frame.shape[1], frame.shape[0])

                    cv2.imshow(self.image_path, frame)
                    cv2.setMouseCallback(self.image_path, CreateGT.mouse_event_handler, gt)
                    key = cv2.waitKey() & 0xFF
                    if key == ord('q'):
                        isRun = False
                        break
                    elif key == ord('n'):
                        pos = min(pos + 1, end - 1)
                    elif key == ord('p'):
                        pos = max(pos - 1, 0)
                else:
                    break

            cv2.destroyAllWindows()
        
        if self.mode == 'cgt_yot':            
            name = os.path.join(path, 'groundtruth_rect.txt')      
            np.savetxt(name, gt.gt.astype(int), delimiter=',', fmt='%d')
        elif self.mode == 'cgt_yolo':
            for i, f in enumerate(files):
                name = f.replace(".png", ".txt").replace(".jpg", ".txt")
                name = os.path.join(path, name)
                np.savetxt(name, [gt.gt[i]], delimiter=',', fmt='%.06f')

    def run(self):
        if self.mode is 'vtoi':
            self.vtoi(self.image_path)
        elif self.mode is 'vyolo':
            self.vyolo(self.image_path)
        elif self.mode is 'vrolo':
            self.vrolo(self.image_path)
        elif self.mode is 'cgt_yot' or self.mode is 'cgt_yolo':
            self.cgt(self.image_path)
        else:
            assert False, 'Error : Unknown mode!!!'

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
            cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]+rect[0]), int(rect[3]+rect[1])), (0, 255, 0), 1)
            cv2.imshow(gt.img_name, img)
            cv2.displayStatusBar(gt.img_name, f'{gt.pos} : {rect}')
            print(gt.pos, rect)
            if gt.mode == 'cgt_yolo': # # (X1, Y1, W, H) ==> (Cx, Cy, W, H)      
                rect[0] = (rect[0] + rect[2]/2)/gt.img_size[0]
                rect[1] = (rect[1] + rect[3]/2)/gt.img_size[1]
                rect[2] = rect[2]/gt.img_size[0]
                rect[3] = rect[3]/gt.img_size[1]

def main(argvs):
    cgt = CreateGT()
    cgt.run()

if __name__=='__main__':
    main('')
