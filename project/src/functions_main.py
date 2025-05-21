#Authors: Timo Michoud, Sina Röllin, Veronika Podliesnova


#Import necessary libraries
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# descriptor params
REF_SIZE    = (128,128)
HOG_PARAMS  = dict(orientations=9,
                   pixels_per_cell=(16,16),
                   cells_per_block=(2,2),
                   block_norm='L2-Hys',
                   feature_vector=True)
LBP_P       = 8
LBP_R       = 1
LBP_METHOD  = 'uniform'
HSV_BINS    = [50,50]   # H & V channels

def classify_group(mean_rgb, mean_hsv, dist):
    """
    Classifies the group of the object based on its mean RGB, its HSV values and its 
    distance to a reference colour.

    Parameters
    ----------
    mean_rgb : tuple
        The mean RGB values.
    mean_hsv : tuple
        The mean HSV values.
    dist : int
        The maximal distance to a reference colour.

    Returns
    -------
    categorical: string    
        The group of the object.
    """

    #White Background
    if mean_hsv[0] < 15 and mean_hsv[1] < 7 and dist < 116 and dist >68:
        return "White Background"
    #Red Jelly Bag
    if mean_rgb[0] >168 and dist< 107 and mean_hsv[0]<8 and mean_hsv[0]> 4 and mean_hsv[1]>28:
        return "Red Jelly Bag"
    #Orange/Blue Book
    if mean_hsv[0]<50 and mean_hsv[1]>27:
        return "Orange/Blue Book"
    #Blue Book
    if dist < 108 and dist> 78 and mean_hsv[0]> 90:
        return "Blue Book"
    #Other
    return "Other"

def rgb_to_hsv(rgb):
    """
    Converts a pixel from RGB color to HSV color space.
    
    Parameters
    ----------
    
    rgb : tuple
        A tuple of three integers representing the RGB color.
        
    Returns
    -------
    hsv : tuple
        A tuple of three integers representing the HSV color.
    """
    color = np.uint8([[rgb]])
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv[0][0]

def create_combined_mask_rgb(image_rgb, color_data, tolerance=(30, 30, 30), kernel_fill=30, kernel_open=25):
    """
    Create a combined mask in RGB color space.
    The mask comprises the returning pixels of the image comprised whithin the given tolerance of at least one of the reference colors
    Closing and opening are applied to the mask to fill the holes in the mask.

    Parameters
    ----------
    image_rgb: numpy.ndarray 
        Input image in RGB format.
    color_data: dict
        Dictionary of reference colors.
    tolerance: tuple
        Tolerance for color matching (R, G, B).

    Returns
    -------
    combined_mask: numpy.ndarray
        The combined mask of the detected colors.
    """
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for _, rgb_colors in color_data.items():
        for rgb in rgb_colors:
            # Define lower and upper bounds for the color with tolerance
            lower = np.array([max(c - t, 0) for c, t in zip(rgb, tolerance)], dtype=np.uint8)
            upper = np.array([min(c + t, 255) for c, t in zip(rgb, tolerance)], dtype=np.uint8)

            mask = cv2.inRange(image_rgb, lower, upper)

            # Combine the mask with the overall mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # # Apply morphological operations to fill small holes and remove noise
    kernel_fill = np.ones((kernel_fill, kernel_fill), np.uint8)
    kernel_open = np.ones((kernel_open, kernel_open), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fill)  # Fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)   # Remove noise

    return combined_mask

def create_combined_mask_hsv(image_bgr, color_data, tolerance=(5, 40, 30), kernel_fill=20, kernel_open=20):
    """
    Creates a combined mask for the given image using the HSV color space.
    The mask comprises the returning pixels of the image comprised whithin the given tolerance of at least one of the reference colors
    Closing and opening are applied to the mask to fill the holes in the mask.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        The input image in BGR format.
    color_data : dict
        A dictionary containing color names and their corresponding RGB values.
    tolerance : tuple
        A tuple of three integers representing the tolerance for hue, saturation, and value.
    kernel_fill : int
        The size of the kernel for morphological closing operation.
    kernel_open : int
        The size of the kernel for morphological opening operation.
    Returns
    -------
    combined_mask : numpy.ndarray
        The combined mask of the detected colors.
    """
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for _, rgb_colors in color_data.items():
        for rgb in rgb_colors:
            hsv = rgb_to_hsv(rgb)
            lower = np.array([
                max(int(hsv[0]) - tolerance[0], 0),
                max(int(hsv[1]) - tolerance[1], 0),
                max(int(hsv[2]) - tolerance[2], 0)
            ], dtype=np.uint8)
            upper = np.array([
                min(int(hsv[0]) + tolerance[0], 255),
                min(int(hsv[1]) + tolerance[1], 255),
                min(int(hsv[2]) + tolerance[2], 255)
            ], dtype=np.uint8)
            mask = cv2.inRange(hsv_image, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    #Morphological operations
    kernel_fill = np.ones((kernel_fill, kernel_fill), np.uint8)
    kernel_open = np.ones((kernel_open, kernel_open), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fill)  # Fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)   # Remove noise

    return combined_mask

def lab_mask(preproc_rgb):
    lab = cv2.cvtColor(preproc_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    ab_mask = cv2.inRange(a, 135, 150)
    mask = ab_mask 

    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # erode
    mask = cv2.erode(mask, kernel, iterations=2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    return mask

def extract_descriptor(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, REF_SIZE)
    # HOG
    h = hog(gray, **HOG_PARAMS)
    # LBP
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, LBP_METHOD)
    lbp_hist,_ = np.histogram(lbp.ravel(), bins=LBP_P+2, range=(0,LBP_P+2))
    lbp_hist    = lbp_hist.astype(float)
    lbp_hist   /= (lbp_hist.sum()+1e-6)
    # HSV hist (H & V channels)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv],[0],None,[HSV_BINS[0]],[0,180]).flatten()
    v_hist = cv2.calcHist([hsv],[2],None,[HSV_BINS[1]],[0,256]).flatten()
    h_hist/= (h_hist.sum()+1e-6)
    v_hist/= (v_hist.sum()+1e-6)
    return np.hstack([h, lbp_hist, h_hist, v_hist])