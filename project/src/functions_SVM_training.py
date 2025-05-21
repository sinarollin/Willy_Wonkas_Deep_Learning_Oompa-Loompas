#Authors: Timo Michoud, Sina Röllin, Veronika Podliesnova


#Import necessary libraries
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


#############################################################
#extract_descriptor
def extract_descriptor(patch, ref_size, HOG_params, LBP_P, LBP_R, LBP_METHOD, HSV_BINS):
    """
    Extract a feature descriptor containing HOG, LBP, hue and value histograms from an image.

    Parameters
    ----------
    patch : np.ndarray
        Input image patch in BGR format.
    ref_size : tuple
        Size (width, height) to resize each image for feature extraction.
    HOG_params : dict
        Dictionnary of parameters for HOG feature
    LBP_P : int
        Number of neighbor points for LBP.
    LBP_R : float
        Radius of circle for LBP.
    LBP_METHOD : str
        Method to compute LBP (e.g., 'uniform', 'default').
    HSV_BINS : list of ints
        Number of bins for the hue and value channel histograms.

    Returns
    -------
    descriptor : np.ndarray
        A 1D array combining HOG (h), LBP (lbp_hist), hue (h_hist) and value (v_hist) features.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, ref_size)
    # HOG
    h = hog(gray, **HOG_params)
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


def train_SVM(X_train,Y_train, c_param):
    """
    Training of an SVM classifier to recognize classes of chololate 
    based on HOG, LBP, hue and value features.
    Saves the trained model as svm_choco_proba.joblib file and returns it.

    Parameters
    ----------
    X_train: np.ndarray
        Array of samples with HOG, LBP, hue and value features.
    Y_train: np.ndarray
        Array of chocolate labels corresponding to X_train samples
    c_param: Float
        Regularization parameter. Must be strictly positive.
        
    Returns
    -------
    clf: Pipeline
        trained SVM classifier
    """
    print("Training calibrated SVM…")
    base = svm.LinearSVC(C=c_param, max_iter=5000)
    clf = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(base, cv=5)   # 5-fold to train probabilities
    )
    clf.fit(X_train, Y_train)
    joblib.dump(clf, 'svm_choco_proba.joblib')

    return clf