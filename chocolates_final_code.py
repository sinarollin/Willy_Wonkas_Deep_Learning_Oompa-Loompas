import os, cv2, json, glob, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import Counter


ANN_JSON    = '2D-on-2D_annotations_combined90.json'
TRAIN_IMG   = '../train'
TEST_IMG    = '../test'
TEST_PRED   = Path('test_pred')
SUB_SAMPLE  = 'sample_submission.csv'
SUB_OUT     = 'submission_classic_new.csv'
PREPROC_DIR = Path('preprocessed')
MASK_DIR    = Path('masks')

os.makedirs(PREPROC_DIR, exist_ok=True)
os.makedirs(MASK_DIR,    exist_ok=True)
os.makedirs(TEST_PRED,   exist_ok=True)


MIN_AREA    = 10000
MORPH_KSIZE = 9
MORPH_KERNEL_SIZE = 9


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


# SVM params
C_PARAM     = 1.0       # regularization



# Color data (RGB tuples)
color_data = {
    "Amandina": [(180, 172, 152), (143, 96, 54), (195, 179, 128)],
    "Arabia": [(101, 61, 38), (50, 30, 23)],
    "Comtesse": [(225, 225, 217), (186, 172, 137)],
    "Crème Brulée": [(197, 188, 159), (148, 75, 32)],
    "Jelly Black": [(34, 23, 21), (70, 83, 125)],
    "Jelly Milk": [(86, 49, 30), (142, 135, 153)],
    "Jelly White": [(202, 193, 162), (228, 229, 223)],
    "Noblesse": [(138, 92, 59), (61, 37, 27)],
    "Noir Authentique": [(67, 37, 27), (118, 82, 58)],
    "Passion au Lait": [(94, 56, 35), (181, 175, 143), (134, 110, 112)],
    "Stracciatella": [(81, 53, 40), (139, 137, 142)],
    "Tentation Noir": [(75, 45, 35), (42, 30, 32), (103, 79, 79)],
    "Triangolo": [(99, 52, 32), (130, 97, 90), (67, 37, 27)],
}


group_tolerances = {
    "White Background": { "rgb":(10,10,10), "hsv":(10,40,40) },
    "Red Jelly Bag":    { "rgb":(15,10,10), "hsv":(5,30,30) },
    "Blue Book":        { "rgb":(10,10,20), "hsv":(5,20,30) },
    "Orange/Blue Book": { "rgb":(18,18,18), "hsv":(10,30,30)},
    "Other":            { "rgb":(25,25,25), "hsv":(10,40,40)},
}
replacement_color = (183,181,184)

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

def preprocess(img_bgr, tol_rgb, tol_hsv):
    """Color‐replace background according to group‐specific tolerances."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_mask = create_combined_mask_rgb(img_rgb, color_data,
                                        tolerance=tol_rgb,
                                        kernel_fill=30, kernel_open=25)
    hsv_mask = create_combined_mask_hsv(img_bgr, color_data,
                                        tolerance=tol_hsv,
                                        kernel_fill=20, kernel_open=20)
    fg = cv2.bitwise_or(rgb_mask, hsv_mask)
    out = img_rgb.copy()
    out[fg == 0] = replacement_color
    return out


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




print("Loading annotations…")
with open(ANN_JSON) as f:
    ann = json.load(f)

# map image filename → list of bboxes + class id
gt = {}
for img in ann['images']:
    fname = img['image']         # 'L1000768.JPG'
    items = []
    for obj in img['annotations']:
        cls_name = obj['class']  # e.g. 'Comtesse'
        # find the index of the class dict whose 'name' matches
        cls_idx = next(
            i for i, c in enumerate(ann['classes'])
            if c['name'] == cls_name
        )
        bb = obj['boundingBox']
        # Round or int-cast the floats to get pixel coordinates
        x, y = int(round(bb['x'])), int(round(bb['y']))
        w, h = int(round(bb['width'])), int(round(bb['height']))
        items.append((cls_idx, x, y, w, h))

    gt[fname] = items
    
X_train, y_train = [], []
print("Extracting descriptors from train bboxes…")
for fname, objects in tqdm(gt.items()):
    img = cv2.imread(os.path.join(TRAIN_IMG, fname))
    for cls,x,y,w,h in objects:
        patch = img[y:y+h, x:x+w]
        desc = extract_descriptor(patch)
        X_train.append(desc)
        y_train.append(cls)

X_train = np.stack(X_train)
y_train = np.array(y_train)




print("Training SVM…")
clf = make_pipeline(StandardScaler(), svm.LinearSVC(C=C_PARAM, max_iter=5000))
clf.fit(X_train, y_train)
joblib.dump(clf, 'svm_choco.joblib')





sample = pd.read_csv(SUB_SAMPLE)

out    = sample.copy()

class_names = [c['name'] for c in ann['classes']]

col_to_idx = {
    col: class_names.index(col)
    for col in sample.columns
    if col != 'id'
}


for idx, row in tqdm(sample.iterrows(), total=len(sample)):
    img_id = int(row['id'])
    fn     = f"L{img_id}.JPG"
    path   = Path(TEST_IMG)/fn
    img    = cv2.imread(str(path))
    if img is None:
        path = Path(TEST_IMG)/f"L{img_id}.jpg"
        img  = cv2.imread(str(path))
    if img is None:
        print("Missing", fn); continue

    # 1) Preprocess (color replace) -------------
    mean_bgr = np.mean(img, axis=(0,1))
    mean_rgb = mean_bgr[::-1]
    dist     = np.linalg.norm(mean_rgb - replacement_color)
    mean_hsv = cv2.cvtColor(
                   np.uint8([[mean_rgb]]),
                   cv2.COLOR_RGB2HSV
               )[0,0]
    group    = classify_group(mean_rgb, mean_hsv, dist)
    tol_rgb  = group_tolerances[group]['rgb']
    tol_hsv  = group_tolerances[group]['hsv']
    
    preproc_path = PREPROC_DIR/fn
    if preproc_path.exists():
        preproc = cv2.cvtColor(cv2.imread(str(preproc_path)), cv2.COLOR_BGR2RGB)
    else:
        preproc = preprocess(img, tol_rgb, tol_hsv)
        # save as BGR
        cv2.imwrite(str(preproc_path),
                    cv2.cvtColor(preproc, cv2.COLOR_RGB2BGR))
        

    mask_path = MASK_DIR/f"{img_id}.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        mask = lab_mask(preproc)
        cv2.imwrite(str(mask_path), mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes  = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA: 
            continue
        elif cv2.contourArea(c) > 700000:
            print("Too big", fn, cv2.contourArea(c))
            continue
        
        x,y,w,h = cv2.boundingRect(c)
        # if one axis is more than 3x the other, skip
        if w > 3*h or h > 3*w:
            continue
        boxes.append((x,y,w,h))

    #print("Found", len(boxes), "pieces in", fn)
    cnts  = Counter()
    vis   = img.copy()
    for (x,y,w,h) in boxes:
        patch = img[y:y+h, x:x+w]
        desc  = extract_descriptor(patch)
        cls   = clf.predict(desc.reshape(1,-1))[0]
        cnts[cls] += 1
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
        name = class_names[cls]
        cv2.putText(vis, name, (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0),2)


    for col, cls_idx in col_to_idx.items():
        out.at[idx, col] = cnts.get(cls_idx, 0)

    cv2.imwrite(str(TEST_PRED/fn), vis)

out.to_csv(SUB_OUT, index=False)
print("Done →", SUB_OUT)
