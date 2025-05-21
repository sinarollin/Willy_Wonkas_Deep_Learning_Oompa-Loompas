#Authors: Timo Michoud, Sina Röllin, Veronika Podliesnova


#Import necessary libraries
import cv2
import numpy as np


# 2) Mask creation functions
def create_rgb_mask(img_rgb, color_data, tol=(15,15,15)):
    """
    Create a mask in RGB color space.The mask comprises the returning pixels of the image comprised 
    within the given tolerance of at least one of the reference colors. 
    
    Parameters
    ----------
    image_rgb: numpy.ndarray 
        Input image in RGB format.
    color_data: dict
        Dictionary of reference colors.
    tol: tuple
        Tolerance for color matching (R, G, B).

    Returns
    -------
    mask: numpy.ndarray
        The mask of the detected colors.
    """
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for colors in color_data.values():
        for c in colors:
            diff = np.abs(img_rgb.astype(int) - np.array(c)[None,None,:])
            mask[np.all(diff <= tol, axis=-1)] = 255
    return mask

def create_hsv_mask(img_bgr, color_data, tol=(5,40,30)):
    """
    Create a mask in HSV color space.The mask comprises the returning pixels of the image comprised 
    within the given tolerance of at least one of the reference colors. 
    
    Parameters
    ----------
    img_bgr: numpy.ndarray 
        Input image in BGR format.
    color_data: dict
        Dictionary of reference colors.
    tol: tuple
        Tolerance for color matching (H, S, V).

    Returns
    -------
    mask: numpy.ndarray
        The mask of the detected colors.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for colors in color_data.values():
        for c in colors:
            bgr = np.uint8([[[c[2],c[1],c[0]]]])
            hsv_ref = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0,0]
            diff = np.abs(hsv.astype(int) - hsv_ref[None,None,:])
            dh = np.minimum(diff[:,:,0], 180 - diff[:,:,0])
            ds, dv = diff[:,:,1], diff[:,:,2]
            mask[(dh<=tol[0])&(ds<=tol[1])&(dv<=tol[2])] = 255
    return mask

def preprocess(img_bgr, color_data, replacement_color):
    """
    Preprocess an image by replacing the background with replacement_color
    
    Parameters
    ----------
    img_bgr: numpy.ndarray
        Input image in BGR format.
    color_data: dict
        Dictionary of reference colors. 
    replacement_color: tuple
        RGB value of the color to replace the background.

    Returns
    -------
    out: numpy.ndarray
        The preprocessed image.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    m_rgb   = create_rgb_mask(img_rgb, color_data)
    m_hsv   = create_hsv_mask(img_bgr, color_data)
    fg_mask = cv2.bitwise_or(m_rgb, m_hsv)
    out     = img_rgb.copy()
    out[fg_mask==0] = replacement_color
    return out

def otsu_mask(preproc_rgb):
    """
    Create a mask for a given image usin Otsu's thresholding method.

    Parameters
    ----------
    preproc_rgb: numpy.ndarray
        Input image in RGB format.
    
    Returns
    -------
    mask: numpy.ndarray
        The Otsu mask for the image.
    """
    gray = cv2.cvtColor(preproc_rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # invert so chocolates=white
    mask = cv2.bitwise_not(m)

    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.erode(mask, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def draw_boxes(img_bgr, mask, min_area):
    """
    Find objects with a contour area larger than MIN_AREA in a masked image.
    Return a copy of the original image with a bounding box of (255,0,255) color around each object found. 

    Parameters
    ----------
    img_bgr: numpy.ndarray
        Input image in BGR format.
    mask: numpy.ndarray
        Input image masked.
        
    Returns
    -------
    vis: numpy.ndarray
        The image with bounding boxes drawn on.
    """
    vis = img_bgr.copy()
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x,y),(x+w,y+h), (255,0,255), 2)
    return vis