import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

def loadIndices(jsondata, indices, asTensor = True):
    '''
    Load the masks corresponding to the indices from the JSON data

    inputs
    jsondata: JSON data as a Python dictionary
    indices: list of indices to load

    outputs
    masks: list of masks as PyTorch tensors
    '''
    masks = []
    for idx in indices:
        mask = mask_utils.decode(jsondata["annotations"][idx]["segmentation"])
        if asTensor:
            mask = torch.tensor(mask, dtype=torch.float32)
        masks.append(mask)
    return masks

def interactiveSelect(image, jsondata):
    '''
    Open an OpenCV window to select masks from the JSON data

    inputs
    image: image as a Numpy array
    jsondata: JSON data as a Python dictionary

    output
    indices: list of indices for corresponding masks
    '''
    indices = []
    clone = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, annotation in enumerate(jsondata["annotations"]):
                mask = mask_utils.decode(annotation["segmentation"])
                #mask = masks[idx]
                if mask[y, x] == 1:
                    if idx not in indices:
                        indices.append(idx)
                    else:
                        indices.remove(idx)
                    display_image = clone.copy()
                    for index in indices:
                        mask = mask_utils.decode(jsondata["annotations"][index]["segmentation"])
                        #mask = masks[index]
                        display_image[mask == 1] = [0, 255, 0]
                    cv2.imshow("image", display_image[:,:,::-1])
                    break
                    # # Display the whole piceces of masks, comment the break
                    g
    cv2.imshow("image", image[:,:,::-1])
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return indices


