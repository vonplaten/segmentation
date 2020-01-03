import os
import numpy as np
from PIL import Image
import colorlover as cl
import cv2

import src.util.transform

def print_images_masks(images: tuple, masks: tuple, printfolder: str, printfile: str):
    '''Prints given images with masks painted on each image

    Images printed are given prefixes 1_, 2_ ..
    Images and masks must be numpy uint8 in format [w,h,c]

    Parameters
    ----------
    images: tuple containing numpy.ndarray of type np.uint8
    masks: tuple containing numpy.ndarray of type np.uint8
    printfolder: str
        Folder to save images in, relative to project root
    printfile: str
        Filename e.g image.png
    '''
    assert(isinstance(images, tuple))
    assert(isinstance(masks, tuple))
    assert(len(images) == len(masks) > 0)
    assert(all(isinstance(x, np.ndarray) for x in images))
    assert(all(isinstance(x, np.ndarray) for x in masks))

    if not os.path.isdir(printfolder):
        os.makedirs(printfolder)
    for i in range(len(images)):
        pil_im = Image.fromarray(getpaintedimage(images[i], masks[i]))
        pil_im.save(os.path.join(printfolder, f"{i}_{printfile}"))

def getpaintedimage(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    '''Paints mask on image

    Masks are painted in different colours automaticly.

    Parameters
    ----------
    image: numpy.ndarray
        The image to be painted. Must be numpy.ndarray of shape (w,h,c) and
        type np.uint8.
    masks: numpy.ndarray
        The mask for painting. Must be numpy.ndarray of shape (w,h,c) and
        type np.uint8.

    Returns
    -------
    numpy.ndarray
        A array representing the painted image in format (w,h,c) of
        type np.uint8
    '''
    result = image.copy()
    masklist = tuple(masks[:, :, i] for i in range(masks.shape[2]))
    num_classes = masks.shape[2]
    colors = cl.scales[f"{num_classes}"]['qual']['Set3']
    labels = np.array(range(1, num_classes + 1))
    palette = dict(zip(labels, np.array(cl.to_numeric(colors))))
    for i in range(num_classes):
        mask_layer = masklist[i]
        color = palette[i + 1]
        contours, _ = cv2.findContours(mask_layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            result = cv2.drawContours(
                image=result,
                contours=contours,
                contourIdx=-1,
                color=color,
                thickness=2)
        if mask_layer.sum() > 0:
            width = image.shape[0]
            scale = 0.4
            cv2.putText(
                img=result,
                text=f"{i+1}",
                org=(int(20 * scale) + (i * 16), int(40 * scale)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=scale,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA)
    return result
