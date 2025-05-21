import os, cv2, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter


from src.functions_SVM_training import *
from src.functions_inference import *


#DEFINE PATHS
ANN_JSON    = '2D-on-2D_annotations_export (9).json'
TRAIN_IMG   = 'dataset_project_iapr2025/train'
TEST_IMG    = 'dataset_project_iapr2025/test'
TEST_PRED   = 'test_pred'
SUB_SAMPLE  = 'dataset_project_iapr2025/sample_submission.csv'
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


########################################################SVM CLASSIFICATION WITH TRAINING DATA########################################################
""" def extract_descriptor(patch):
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
    return np.hstack([h, lbp_hist, h_hist, v_hist]) """

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

clf = train_SVM(X_train,y_train)
""" 
print("Training SVM…")
clf = make_pipeline(StandardScaler(), svm.LinearSVC(C=C_PARAM, max_iter=5000))
clf.fit(X_train, y_train)
joblib.dump(clf, 'svm_choco.joblib')

print("Training calibrated SVM…")
base = svm.LinearSVC(C=C_PARAM, max_iter=5000)
clf = make_pipeline(
    StandardScaler(),
    CalibratedClassifierCV(base, cv=5)   # 5-fold to train probabilities
)
clf.fit(X_train, y_train)
joblib.dump(clf, 'svm_choco_proba.joblib') """


########################################################INFERENCE OF TEST DATA########################################################
########################PREPROCESSING FUNCTIONS#####################
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
replacement_color = (183, 181, 184)  # RGB

""" # 2) Mask creation functions
def create_rgb_mask(img_rgb, color_data, tol=(15,15,15)):
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for colors in color_data.values():
        for c in colors:
            diff = np.abs(img_rgb.astype(int) - np.array(c)[None,None,:])
            mask[np.all(diff <= tol, axis=-1)] = 255
    return mask

def create_hsv_mask(img_bgr, color_data, tol=(5,40,30)):
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

def preprocess(img_bgr):
    # Replace background by replacement_color.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    m_rgb   = create_rgb_mask(img_rgb, color_data)
    m_hsv   = create_hsv_mask(img_bgr,   color_data)
    fg_mask = cv2.bitwise_or(m_rgb, m_hsv)
    out     = img_rgb.copy()
    out[fg_mask==0] = replacement_color
    return out

def otsu_mask(preproc_rgb):
    # Compute Otsu mask on preprocessed RGB.
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

def draw_boxes(img_bgr, mask):
    vis = img_bgr.copy()
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x,y),(x+w,y+h), (255,0,255), 2)
    return vis """

########################INFERENCE#####################
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
    preproc_path = PREPROC_DIR/fn
    if preproc_path.exists():
        preproc = cv2.cvtColor(cv2.imread(str(preproc_path)), cv2.COLOR_BGR2RGB)
    else:
        preproc = preprocess(img, color_data, replacement_color)
        # save as BGR
        cv2.imwrite(str(preproc_path),
                    cv2.cvtColor(preproc, cv2.COLOR_RGB2BGR))

    # 2) Otsu mask  -----------------------------
    mask_path = MASK_DIR/f"{img_id}.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        mask = otsu_mask(preproc)
        cv2.imwrite(str(mask_path), mask)

    # Find contours → list of (mask, bbox)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes  = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA: 
            continue
        x,y,w,h = cv2.boundingRect(c)
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

    # 3) Draw boxes on original -----------------
    #vis   = draw_boxes(img, mask)

    # 4) Count & fill submission ---------------
    #cnts  = Counter()
    #cnts.update(1 for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #            if cv2.contourArea(c) >= MIN_AREA)
    #for col, cls_i in col_to_idx.items():
    #    out.at[idx, col] = cnts.get(cls_i, 0)

    for col, cls_idx in col_to_idx.items():
        out.at[idx, col] = cnts.get(cls_idx, 0)

    # 5) Save visualization ---------------------
    cv2.imwrite(str(TEST_PRED/fn), vis)

# 6) Save submission --------------------------
out.to_csv(SUB_OUT, index=False)
print("Done →", SUB_OUT)
 