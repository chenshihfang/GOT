import matplotlib.pyplot as plt
import numpy as np
import cv2


# def heat_show(cls_score, name, patch_size=18):
def heat_show(cls_score, name, patch_size=22):
#     patch_size = 18
    size = patch_size*4

    try :
            heatmap = cv2.resize(cls_score, (size,size), interpolation=cv2.INTER_AREA)
    except:
            heatmap = cv2.resize(cls_score.cpu().detach().numpy(), (size,size), interpolation=cv2.INTER_AREA)

    heatmapshow = None
    heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_VIRIDIS)
    
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, heatmapshow)
    cv2.waitKey(1)
