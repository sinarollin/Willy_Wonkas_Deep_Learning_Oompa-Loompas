#Authors


#Import necessary libraries
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from collections import Counter


def extract_chocolate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    #background = median of border pixels
    border_pixels = np.hstack([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    bg_val = int(np.median(border_pixels))

    diff = cv2.absdiff(gray, np.full_like(gray, bg_val))

    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # pick largest by area
    main_cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(main_cnt)
    x, y, w_box, h_box = cv2.boundingRect(hull)

    hull_mask = np.zeros_like(gray)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
    extracted = cv2.bitwise_and(image, image, mask=hull_mask)

    return extracted, (x, y, w_box, h_box)




def build_ref_descriptors(ref_dir, 
                          REF_SIZE=(64, 64),
                          HOG_ORIENT=9,
                          PIXELS_CELL=(8, 8),
                          CELLS_BLOCK=(2, 2),
                          LBP_P=8,
                          LBP_R=1,
                          LBP_METHOD='uniform',
                          LBP_N_BINS=10,
                          H_BINS=16,
                          V_BINS=16):
    """
    Build reference descriptors for the images in the given directory.
    The descriptors are computed using HOG, LBP, and HSV histograms.
    Parameters
    ----------
    ....
    """
    
    refs = {}
    for fn in sorted(os.listdir(ref_dir)):
        if not fn.lower().endswith(('.png','.jpg','.jpeg')): continue
        cls = os.path.splitext(fn)[0]
        img = cv2.imread(os.path.join(ref_dir,fn))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize to REF_SIZE
        gray_r = cv2.resize(gray, REF_SIZE)
        # HOG
        hdesc = hog(gray_r, orientations=HOG_ORIENT,
                    pixels_per_cell=PIXELS_CELL, cells_per_block=CELLS_BLOCK,
                    block_norm='L2-Hys', feature_vector=True)
        # LBP
        lbp = local_binary_pattern(gray_r, LBP_P, LBP_R, LBP_METHOD)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0,LBP_N_BINS))
        lbp_hist = lbp_hist.astype(float); lbp_hist /= lbp_hist.sum()+1e-6
        # HSV hist
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv],[0],None,[H_BINS],[0,180]).flatten()
        v_hist = cv2.calcHist([hsv],[2],None,[V_BINS],[0,256]).flatten()
        h_hist /= h_hist.sum()+1e-6; v_hist /= v_hist.sum()+1e-6
        # concat & L2 normalize
        desc = np.hstack([hdesc, lbp_hist, h_hist, v_hist])
        desc /= np.linalg.norm(desc)+1e-6
        refs[cls] = desc
        print(f"Ref '{cls}': descriptor length {len(desc)}")
    return refs




def segment_pieces(img, MIN_AREA=10000, MORPH_KSIZE=9):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    border = np.hstack([gray[0],gray[-1],gray[:,0],gray[:,-1]])
    bg = int(np.median(border))
    diff = cv2.absdiff(gray, np.full_like(gray,bg))
    _,mask = cv2.threshold(diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_KSIZE,MORPH_KSIZE))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kern,iterations=3)
    mask = cv2.dilate(mask,kern,iterations=2)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pieces=[]
    for c in cnts:
        if cv2.contourArea(c)<MIN_AREA: continue
        x,y,w,h = cv2.boundingRect(c)
        m = np.zeros_like(gray); cv2.drawContours(m,[c],-1,255,-1)
        pieces.append((m,(x,y,w,h)))
    return pieces


def classify_and_draw(img, ref_descs, classes,
                      REF_SIZE=(64, 64),
                      HOG_ORIENT=9,
                      PIXELS_CELL=(8, 8),
                      CELLS_BLOCK=(2, 2),
                      LBP_P=8,
                      LBP_R=1,
                      LBP_METHOD='uniform',
                      LBP_N_BINS=10,
                      H_BINS=16,
                      V_BINS=16,
                      SIM_THRESH=0.5):
    
    pieces = segment_pieces(img)
    cnts   = Counter()
    vis    = img.copy()
    for mask,(x,y,w,h) in pieces:
        patch = img[y:y+h,x:x+w]
        gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        gray_r = cv2.resize(gray, REF_SIZE)
        # HOG+LBP
        hdesc = hog(gray_r, orientations=HOG_ORIENT,
                    pixels_per_cell=PIXELS_CELL, cells_per_block=CELLS_BLOCK,
                    block_norm='L2-Hys', feature_vector=True)
        lbp = local_binary_pattern(gray_r, LBP_P, LBP_R, LBP_METHOD)
        lbp_hist,_ = np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0,LBP_N_BINS))
        lbp_hist = lbp_hist.astype(float); lbp_hist/=lbp_hist.sum()+1e-6
        # HSV
        hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv],[0],None,[H_BINS],[0,180]).flatten()
        v_hist = cv2.calcHist([hsv],[2],None,[V_BINS],[0,256]).flatten()
        h_hist/=h_hist.sum()+1e-6; v_hist/=v_hist.sum()+1e-6
        desc = np.hstack([hdesc, lbp_hist, h_hist, v_hist])
        desc/=np.linalg.norm(desc)+1e-6

        # nearest neighbor
        best_cls, best_sim = None, -1
        for cls, rdesc in ref_descs.items():
            sim = float(np.dot(desc, rdesc))
            if sim>best_sim:
                best_sim, best_cls = sim, cls

        if best_sim>=SIM_THRESH:
            cnts[best_cls]+=1
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(vis,f"{best_cls}:{best_sim:.2f}",(x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)

    # zero-fill
    for c in classes:
        cnts.setdefault(c,0)
    return dict(cnts), vis