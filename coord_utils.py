import numpy as np
import torch

class coord_utils:
    @staticmethod
    def location_to_probability_map(size, loc):
        # loc is not normalized location
        promap_vec = torch.zeros([size, size], dtype=torch.float32)
        try:
            conf = loc[4]
        except IndexError:
            conf = 1.0

        [x1, y1, x2, y2] = [(loc[0]*size).int(), (loc[1]*size).int(), ((loc[0]+loc[2])*size).int(), ((loc[1]+loc[3])*size).int()]
        if x1 == x2: x2 += 1
        if y1 == y2: y2 += 1
        
        x1 = x1.clamp(0, size)
        y1 = y1.clamp(0, size)
        x2 = x2.clamp(0, size)
        y2 = y2.clamp(0, size)

        for y in range(y1, y2): 
            for x in range(x1, x2):                       
                promap_vec[y][x] = conf 
        return promap_vec

    @staticmethod
    def locations_to_probability_maps(size, locs):
        pms = []
        for loc in locs:
            pm = coord_utils.location_to_probability_map(size, loc)
            pms.append(pm.view(-1))
        
        return torch.stack(pms, dim=0)

    @staticmethod
    def probability_map_to_location(size, pmap):
        # probability map to location
        pmap = pmap.view(size, size)

        xlist = []
        ylist = []
        for y in range(size):
            for x in range(size):
                if(pmap[y][x] >= 0.5):
                    xlist.append(x+0.5)
                    ylist.append(y+0.5)

        if len(xlist) == 0 or len(ylist) == 0:
            return torch.zeros(4, dtype=torch.float32)

        ax = np.array(xlist)
        ay = np.array(ylist)
        xmean = ax.mean()
        ymean = ay.mean()

        k = 2.5 #np.sqrt(2)
        w = ax.std() * k + 0.5
        h = ay.std() * k + 0.5

        x1 = xmean - w/2.
        y1 = ymean - h/2.

        loc = torch.tensor([x1/size, y1/size, w/size, h/size], dtype=torch.float32)
        return loc

    @staticmethod
    def normal_to_location(wid, ht, location):
        # Normalized location to coordinate
        wid *= 1.0
        ht *= 1.0
        location[0] *= wid
        location[1] *= ht
        location[2] *= wid
        location[3] *= ht
        return location

    @staticmethod
    def location_to_normal(wid, ht, location):
        # Coordinates to normalized location
        wid *= 1.0
        ht *= 1.0
        location[0] /= wid
        location[1] /= ht
        location[2] /= wid
        location[3] /= ht
        return location

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou