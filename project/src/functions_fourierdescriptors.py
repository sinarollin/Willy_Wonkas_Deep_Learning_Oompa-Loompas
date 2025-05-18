#Authors: Timo Michoud, Sina Röllin, Veronika Podliesnova


#Import necessary libraries
import numpy as np
import os
import cv2
import pandas as pd


def get_fourier_descriptors(contour, k=5, normalize=True):
    
    cnt = contour.squeeze()
    if cnt.ndim != 2 or cnt.shape[1] != 2:
        raise ValueError("Contour shape is invalid.")

    complex_cnt = cnt[:, 0] + 1j * cnt[:, 1]
    fourier_desc = np.fft.fft(complex_cnt)

    if normalize:
        fourier_desc[0] = 0  # remove translation
        fourier_desc /= np.abs(fourier_desc[1])  # scale invariance
        fourier_desc = np.abs(fourier_desc)  # rotation invariance

    return fourier_desc[:k]

def load_reference_hists(ref_dir, HIST_BINS=16, HIST_CHANNELS=[0, 1], HIST_RANGES=[0, 180, 0, 256]):
    refs = {}
    
    for fn in sorted(os.listdir(ref_dir)):
        if not fn.lower().endswith('.png'):
            continue
        cls_name = os.path.splitext(fn)[0]
        img = cv2.imread(os.path.join(ref_dir, fn))
        if img is None:
            raise IOError(f"Could not load {fn}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # full‐patch hist (we assume tight crops)
        hist = cv2.calcHist([hsv], HIST_CHANNELS, None, HIST_BINS, HIST_RANGES)
        hist = cv2.normalize(hist, hist).flatten()
        #ref_hists[cls_name] = hist
        
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        # 3. Create mask for green
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 4. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        F_contour = get_fourier_descriptors(contour)

        refs[cls_name] = (hist, F_contour)

        print(f"Loaded reference '{cls_name}'")
    return refs


def segment_pieces(img_bgr, MIN_AREA=1000, MAX_AREA=50000, MORPH_KSIZE=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    border = np.hstack([gray[0,:], gray[-1,:], gray[:,0], gray[:,-1]])
    bg_val = int(np.median(border))

    diff = cv2.absdiff(gray, np.full_like(gray, bg_val))
    _, mask = cv2.threshold(diff, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=3)
    mask = cv2.dilate(mask, kern, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    pieces = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        
        x, y, w_box, h_box = cv2.boundingRect(c)
        piece_mask = np.zeros_like(gray)
        cv2.drawContours(piece_mask, [c], -1, 255, thickness=-1)
        pieces.append((piece_mask, (x, y, w_box, h_box),c))
    return pieces


def classify_piece_by_hist(patch_bgr, patch_mask, ref_hists, HIST_BINS=16, HIST_CHANNELS=[0, 1], HIST_RANGES=[0, 180, 0, 256], HIST_METHOD=cv2.HISTCMP_CORREL):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], HIST_CHANNELS, patch_mask,
                        HIST_BINS, HIST_RANGES)
    hist = cv2.normalize(hist, hist).flatten()

    best_cls, best_score = None, -1
    for cls, (ref_hist, _) in ref_hists.items():
        score = cv2.compareHist(ref_hist, hist, HIST_METHOD)
        if score > best_score:
            best_score, best_cls = score, cls
    return best_cls, best_score

def compare_fourier_descriptors(F1, F2, method="weighted_euclidean"):
    # Ensure both have the same length
    min_len = min(len(F1), len(F2))
    F1 = F1[:min_len]
    F2 = F2[:min_len]

    if method == "weighted_euclidean":
        # Define weights: more weight for lower frequencies
        weights = np.linspace(1.0, 0.1, min_len)  # or exponential decay, etc.
        diff = F1 - F2
        score = np.sqrt(np.sum(weights * np.abs(diff)**2))
        return score

    elif method == "cosine":
        # Optional alternative: cosine similarity
        norm1 = np.linalg.norm(F1)
        norm2 = np.linalg.norm(F2)
        score = 1 - np.dot(F1, F2.conj()).real / (norm1 * norm2)
        return score

    else:
        raise ValueError("Unknown comparison method")

def classify_piece_by_F_desc(cont_F, ref_Fs):
    #F_cont = get_fourier_descriptors(cont_F)

    best_cls, best_score = None, 1000000

    for cls, (_, ref_F) in ref_Fs.items():
        # Ensure same length
        min_len = min(len(cont_F), len(ref_F))
        d_cont = cont_F[:min_len]
        d_ref = ref_F[:min_len]

        # Use Euclidean distance
        #score = np.linalg.norm(d_cont - d_ref)
        
        score = compare_fourier_descriptors(d_cont,d_ref)
        # Use cosine similarity
        # score = 1 - np.dot(d_cont, d_ref) / (np.linalg.norm(d_cont) * np.linalg.norm(d_ref))


        if score < best_score:
            best_score, best_cls = score, cls
    return best_cls, best_score

def combined_classifier(cont_F, patch_bgr, patch_mask, refs, HIST_BINS=16, HIST_CHANNELS=[0, 1], HIST_RANGES=[0, 180, 0, 256], HIST_METHOD=cv2.HISTCMP_CORREL):
    
    best_cls, best_score = None, -1

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], HIST_CHANNELS, patch_mask,
                        HIST_BINS, HIST_RANGES)
    hist = cv2.normalize(hist, hist).flatten()
    rows = []

    for cls, (ref_hist, ref_F) in refs.items():
        score_H = cv2.compareHist(ref_hist, hist, HIST_METHOD)
        
        if score_H < 0.1:
            continue

        # Ensure same length
        min_len = min(len(cont_F), len(ref_F))
        d_cont = cont_F[:min_len]
        d_ref = ref_F[:min_len]

        # Use Euclidean distance
        #score = np.linalg.norm(d_cont - d_ref)
        
        score_F = compare_fourier_descriptors(d_cont,d_ref)
        label = f"{cls} (Hist:{score_H:.2f}, Fourier:{score_F:.2f})"
        
        #print(label)
        
        # Use cosine similarity
        # score = 1 - np.dot(d_cont, d_ref) / (np.linalg.norm(d_cont) * np.linalg.norm(d_ref))

        """ comb_score = (10*score_H)+(1/score_F)

        if comb_score > best_score:
            best_score, best_cls = comb_score, cls """
        

        rows.append({
            'class': cls,
            'score_H': score_H,
            'score_F': score_F
        })

    # Create DataFrame of scores
    score_df = pd.DataFrame(rows)

    if score_df.empty:
        return None, 0, score_df
    
    # Normalize scores
    # Higher score_H is better; lower score_F is better
    score_df['norm_H'] = (score_df['score_H'] - score_df['score_H'].min()) / (score_df['score_H'].max() - score_df['score_H'].min() + 1e-6)
    score_df['norm_F'] = 1 - (score_df['score_F'] - score_df['score_F'].min()) / (score_df['score_F'].max() - score_df['score_F'].min() + 1e-6)

    # Combine normalized scores with weights (adjust as needed)
    score_df['combined'] = 0.35 * score_df['norm_H'] + 0.65 * score_df['norm_F']

    # Find best class
    best_row = score_df.loc[score_df['combined'].idxmax()]
    best_cls = best_row['class']
    best_score = best_row['combined']    
    


    return best_cls, best_score, score_df


def old_combined_classifier(cont_F, patch_bgr, patch_mask, refs, HIST_BINS=16, HIST_CHANNELS=[0, 1], HIST_RANGES=[0, 180, 0, 256], HIST_METHOD=cv2.HISTCMP_CORREL):
    best_cls, best_score = None, -1

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], HIST_CHANNELS, patch_mask,
                        HIST_BINS, HIST_RANGES)
    hist = cv2.normalize(hist, hist).flatten()

    for cls, (ref_hist, ref_F) in refs.items():
        score_H = cv2.compareHist(ref_hist, hist, HIST_METHOD)
        
        # Ensure same length
        min_len = min(len(cont_F), len(ref_F))
        d_cont = cont_F[:min_len]
        d_ref = ref_F[:min_len]

        # Use Euclidean distance
        #score = np.linalg.norm(d_cont - d_ref)
        
        score_F = compare_fourier_descriptors(d_cont,d_ref)
        label = f"{cls} (Hist:{score_H:.2f}, Fourier:{score_F:.2f})"
        
        #print(label)
        
        # Use cosine similarity
        # score = 1 - np.dot(d_cont, d_ref) / (np.linalg.norm(d_cont) * np.linalg.norm(d_ref))

        comb_score = (score_H)*(1/score_F)

        if comb_score > best_score:
            best_score, best_cls = comb_score, cls

    return best_cls, best_score