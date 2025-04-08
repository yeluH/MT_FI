# Functions for mtp_yl
# Version update time: 2025/04/07

import os 
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
import shapely
from shapely.geometry import Polygon
# import shapely.plotting
import pandas as pd
from PIL import Image as ima
import warnings
from pathlib import Path
import random
from scipy.stats import entropy

# For function featureselection()
from sklearn import metrics
import statsmodels.api as sm
import statsmodels.tools as tools
from sklearn import linear_model
from mlxtend.feature_selection import SequentialFeatureSelector

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Add function def show_anns (borrowed from one of the Jupyter notebooks).
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    # polygons = []
    # color = []
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    # for ann in sorted_anns:
    #     m = ann['segmentation']
    #     img = np.ones((m.shape[0], m.shape[1], 3))
    #     color_mask = np.random.random((1, 3)).tolist()[0]
    #     for i in range(3):
    #         img[:,:,i] = color_mask[i]
    #     ax.imshow(np.dstack((img, m*0.35)))

def convert_mask(masks):
    mb01 = []
    mb02 = []
    mb = []
    n = len(masks)
    for i in range(0, n):
        mb01.append(masks[i]['segmentation'])
    for i in range(0, n):
        mb02.append(mb01[i]*1)
    for i in range(0, n):
        mb.append(mb02[i].astype('uint8'))
    return mb

def output_mask(im):
    im1 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    mask_r = mask_generator.generate(im1)
    mask_c = convert_mask(mask_r)
    return mask_c

# Function to find and plot filtered contours (find-filter-plot-contour)
def ffpcontour(image, mask, i):
    image_masked = cv2.bitwise_and(image,image,mask = mask[i])
    assert image is not None, "image file could not be read, check with os.path.exists()"
    assert mask is not None, "mask file could not be read, check with os.path.exists()"
    # imgray = cv2.cvtColor(image_masked, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    ret, thresh = cv2.threshold((mask[i]*255), 127, 255, 0)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        contour_f = contours
    else:
        contour_f = []
        for i in range(0, len(contours)):
            # print(i, "len", len(contours[i]))
            if len(contours[i]) > 80:
                contour_f.append(contours[i])
    # print("filtered", "len", len(contour_f), contour_f)
    # Plotting the filtered contour
    # -1 is the contourIdx, (0,255,0) is color, 3 is the thickness
    # print("raw",len(contours))
    # print("filtered", len(contour_f))
    img_con = cv2.drawContours(image_masked, contour_f, -1, (0,255,0), 3) 
    plt.figure(figsize = (15,15))
    plt.imshow(img_con)
    plt.axis('on')
    plt.show
    return contour_f

# Updated Function
## Filtering the masks which are too small (<=5) for contour feature/properties calculation
# Function to find and plot filtered contours (find-filter-plot-contour)
# Run for each mask of each image
def ffpcontour_noplot(image, mask, i):
    assert image is not None, "image file could not be read, check with os.path.exists()"
    assert mask is not None, "mask file could not be read, check with os.path.exists()"
    # imgray = cv2.cvtColor(image_masked, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    ret, thresh = cv2.threshold((mask[i]*255), 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ll = [] # length list
    for i in range(0, len(contours)):
          ll.append(len(contours[i]))
    maxl = max(ll)
    maxindex = ll.index(maxl)
    if (len(contours) == 1) and (maxl >= 6):
        contour_f = contours
    elif (maxl>=80) :
        contour_f = []
        for i in range(0, len(contours)):
        # print(i, "len", len(contours[i]))
            if (len(contours[i]) >= 80):
                contour_f.append(contours[i])
            else:
                contour_f = contour_f
    elif (maxl >= 6):
        contour_f = []
        contour_f.append(contours[maxindex])
    else: 
        contour_f = []
    return contour_f


# Updated 01022024 Add location coordinates of extreme points
# Updated 08022024 Add orientation(in fitEllipse) and angle of rotation(in rotated rectangle)
def cgr(contour):
    assert contour is not None, "image file could not be read, check with os.path.exists()"
    c = contour
    isconvex = cv2.isContourConvex(c) # Checking convexity
    (x,y), (w,h), ar = cv2.minAreaRect(c) # Rotated rectangle with minimum area
    ar_r = ar # Angle of rotation in rotated rectangle
    M = cv2.moments(c) # Moments
    area = cv2.contourArea(c) # Area 
    if (M['m00'] != 0):
        cx = int(M['m10']/M['m00']) # Centroid
        cy = int(M['m01']/M['m00'])
    else:
        cx = x
        cy = y
    xs,ys,ws,hs = cv2.boundingRect(c) # Straight bounding rectangle
    aspect_ratio_wh_s = float(ws)/hs # Aspect ratio
    extent_s = float(area)/(ws*hs) # Extent
    hull = cv2.convexHull(c) # Solidity
    hull_area = cv2.contourArea(hull)
    if (hull_area != 0):
        solidity = float(area)/hull_area
    else:
        solidity = 0
    aspect_ratio_wh = float(w)/h  # Aspect ratio
    extent = float(area)/(w*h) # Extent
    (xe,ye),(MA,ma),ae = cv2.fitEllipse(c)
    ar_e = ae # Angle of rotated ellipse, orientation
    ed = np.sqrt(4*area/np.pi) # Equivalent Diameter
    ratio_ell = float(ma)/MA
    perimeter = cv2.arcLength(c, True) # Arclength
    p_centroid = np.array([float(cx), float(cy)])
    p_masscenter = np.array([float(x), float(y)])
    # for the following two variable, positive (+1) = inside, negative (-1) = outside, or zero (0) = on an edge
    is_cen_inside = cv2.pointPolygonTest(c, p_centroid, False) # Checking if centroid is inside
    is_mce_inside = cv2.pointPolygonTest(c, p_masscenter, False) # Checking if mass center is inside
    leftmost = tuple(c[c[:,:,0].argmin()][0])
    rightmost = tuple(c[c[:,:,0].argmax()][0])
    topmost = tuple(c[c[:,:,1].argmin()][0])
    bottommost = tuple(c[c[:,:,1].argmax()][0])
    leftm = leftmost[0]
    rightm = rightmost[0]
    topm = topmost[1]
    bottomm = bottommost[1]
    return {
        'isconvex': isconvex,
        'area': area,
        'aspect_ratio_wh_s': aspect_ratio_wh_s,
        'extent_s': extent_s,
        'solidity': solidity,
        'aspect_ratio_wh': aspect_ratio_wh,
        'extent': extent,
        'orien_rre': ar_r,
        'orien_ell': ar_e,
        'ed': ed,
        'ratio_ell': ratio_ell,
        'perimeter': perimeter,
        'is_cen_inside': is_cen_inside,
        'is_mce_inside': is_mce_inside,
        'leftm': leftm,
        'rightm': rightm,
        'topm': topm,
        'bottomm': bottomm
    }



# Updated, differentiate isconvex and is_cen_inside
# Updated 01022024, add location coordinates of extreme points
# Updated 08022024, add orientation and angle of rotation
def csga(contours):
    assert contours is not None, "image file could not be read, check with os.path.exists()"
    if len(contours) == 1:
        ga = cgr(contours[0])
    else:
        gal = []
        for i in range(0, (len(contours)-1)):
            gal.append(cgr(contours[i]))
        iscon = []
        al = []
        asps = []
        exts = []
        sol = []
        asp = []
        ext = []
        anr = []
        ane = []
        ed = []
        rate = []
        per = []
        iscen = []
        ismce = []
        lm = []
        rm = []
        tm = []
        bm = []
        for i in range(0, len(gal)):
            iscon.append(gal[0]['isconvex'])
            al.append(gal[0]['area'])
            asps.append(gal[0]['aspect_ratio_wh_s'])
            exts.append(gal[0]['extent_s'])
            sol.append(gal[0]['solidity'])
            asp.append(gal[0]['aspect_ratio_wh'])
            ext.append(gal[0]['extent'])
            anr.append(gal[0]['orien_rre'])
            ane.append(gal[0]['orien_ell'])
            ed.append(gal[0]['ed'])
            rate.append(gal[0]['ratio_ell'])
            per.append(gal[0]['perimeter'])
            iscen.append(gal[0]['is_cen_inside'])
            ismce.append(gal[0]['is_mce_inside'])
            lm.append(gal[0]['leftm'])
            rm.append(gal[0]['rightm'])
            tm.append(gal[0]['topm'])
            bm.append(gal[0]['bottomm'])
        isconvex = np.all(iscon)
        area = np.mean(al, axis = 0)
        aspect_ratio_wh_s = np.mean(asps, axis = 0)
        extent_s = np.mean(exts, axis = 0)
        solidity = np.mean(sol, axis = 0)
        aspect_ratio_wh = np.mean(asp, axis = 0)
        extent = np.mean(ext, axis = 0)
        ar_r = np.mean(anr, axis = 0)
        ar_e = np.mean(ane, axis = 0)
        ed = np.mean(ed, axis = 0)
        ratio_ell = np.mean(rate, axis = 0)
        perimeter = np.mean(per, axis = 0)
        is_cen_inside = np.mean(iscen, axis = 0)
        is_mce_inside = np.mean(ismce, axis = 0)
        leftm = np.min(lm, axis = 0)
        rightm = np.max(rm, axis = 0)
        topm = np.min(tm, axis = 0)
        bottomm = np.max(bm, axis = 0)
        ga = {
            'isconvex': isconvex,
            'area': area,
            'aspect_ratio_wh_s': aspect_ratio_wh_s,
            'extent_s': extent_s,
            'solidity': solidity,
            'aspect_ratio_wh': aspect_ratio_wh,
            'extent': extent,
            'orien_rre': ar_r,
            'orien_ell': ar_e,
            'ed': ed,
            'ratio_ell': ratio_ell,
            'perimeter': perimeter,
            'is_cen_inside': is_cen_inside,
            'is_mce_inside': is_mce_inside,
            'leftm': leftm,
            'rightm': rightm,
            'topm': topm,
            'bottomm': bottomm
        }
    return ga

# Updated 01022024 - Distance calculation between RGB points to gray line
def lineseg_dist(p, a, b):
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])
    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))

# Updated 01022024 - Obtain mean and standard deviation of rgb color distances
def coldistance(mig, mib, mir):
    cdmean = []
    cdstd = []
    for i in range(0,len(mib)):
        cg = mig[i]
        cr = mir[i]
        cb = mib[i]
        diss = []
        t2 = np.array([0,0,0])
        t3 = np.array([255,255,255])
        for j in range(0, len(cg)):
            diss.append(lineseg_dist(np.array([cg[j], cr[j], cb[j]]), t2, t3))
        cdmean.append(np.mean(diss))
        cdstd.append(np.std(diss))
    return cdmean, cdstd



# Updated - Remove the empty contour
# Updated - Convert true/false to 1/0
# Updated 01022024 - Add location coordinate value of extreme points
# Updated 01022024 - Add color 25% and 75% quantile values
# Updated 08022024 - Add orientation and angle of rotation
# mask file mf
def feature_summary(image, mf):
    # Generate a data frame for masks and attributes
    df = pd.DataFrame()
    df['mask'] = range(1, (len(mf)+1))
    df = df.assign(gmedian = None, rmedian = None, bmedian = None,
                   gmean = None, rmean = None, bmean = None,
                   gstd = None, rstd = None, bstd = None,
                   gq25 = None, gq75 = None, rq25 = None,
                   rq75 = None, bq25 = None, bq75 = None,
                   cdmean = None, cdstd = None,
                   isconvex = None, area = None, aspect_ratio_wh_s = None,
                   extent_s = None, solidity = None, aspect_ratio_wh = None,
                   extent = None, orien_rre = None, orien_ell = None,
                   ed = None, ratio_ell = None,
                   perimeter = None, is_cen_inside = None, is_mce_inside = None,
                   leftm = None, rightm = None, topm = None, 
                   bottomm = None)
    mm = [] # masked image
    for i in range(0, len(mf)):
        mm.append(cv2.bitwise_and(image, image, mask = mf[i]))
    mib = []
    mig = []
    mir = []
    for i in range(0, len(mm)):
        mib.append((mm[i][:,:,0])[np.where(((mm[i][:,:,0]) != 0) | ((mm[i][:,:,1]) != 0) | ((mm[i][:,:,2]) != 0))])
        mig.append((mm[i][:,:,1])[np.where(((mm[i][:,:,0]) != 0) | ((mm[i][:,:,1]) != 0) | ((mm[i][:,:,2]) != 0))])
        mir.append((mm[i][:,:,2])[np.where(((mm[i][:,:,0]) != 0) | ((mm[i][:,:,1]) != 0) | ((mm[i][:,:,2]) != 0))])
    cm, cs = coldistance(mig, mib, mir)
    for i in range(0, len(mm)):
        df.at[i, 'bmean'] = np.mean(mib[i], axis = 0)
        df.at[i, 'gmean'] = np.mean(mig[i], axis = 0)
        df.at[i, 'rmean'] = np.mean(mir[i], axis = 0)
        df.at[i, 'bmedian'] = np.median(mib[i], axis = 0)
        df.at[i, 'gmedian'] = np.median(mig[i], axis = 0)
        df.at[i, 'rmedian'] = np.median(mir[i], axis = 0)
        df.at[i, 'bstd'] = np.std(mib[i], axis = 0)
        df.at[i, 'gstd'] = np.std(mig[i], axis = 0)
        df.at[i, 'rstd'] = np.std(mir[i], axis = 0)
        df.at[i, 'bq25'] = np.quantile(mib[i], 0.25, axis = 0)
        df.at[i, 'bq75'] = np.quantile(mib[i], 0.75, axis = 0)
        df.at[i, 'gq25'] = np.quantile(mig[i], 0.25, axis = 0)
        df.at[i, 'gq75'] = np.quantile(mig[i], 0.75, axis = 0)
        df.at[i, 'rq25'] = np.quantile(mir[i], 0.25, axis = 0)
        df.at[i, 'rq75'] = np.quantile(mir[i], 0.75, axis = 0)
        df.at[i, 'cdmean'] = cm[i]
        df.at[i, 'cdstd'] = cs[i]
    for i in range(0, len(mf)):
        con = ffpcontour_noplot(image, mf, i)
        if len(con) != 0 :
            df.at[i, 'isconvex'] = csga(con)['isconvex']
            df.at[i, 'area'] = csga(con)['area']
            df.at[i, 'aspect_ratio_wh_s'] = csga(con)['aspect_ratio_wh_s']
            df.at[i, 'extent_s'] = csga(con)['extent_s']
            df.at[i, 'solidity'] = csga(con)['solidity']
            df.at[i, 'aspect_ratio_wh'] = csga(con)['aspect_ratio_wh']
            df.at[i, 'extent'] = csga(con)['extent']
            df.at[i, 'ed'] = csga(con)['ed']
            df.at[i, 'orien_rre'] = csga(con)['orien_rre']
            df.at[i, 'orien_ell'] = csga(con)['orien_ell']
            df.at[i, 'ratio_ell'] = csga(con)['ratio_ell']
            df.at[i, 'perimeter'] = csga(con)['perimeter']
            df.at[i, 'is_cen_inside'] = csga(con)['is_cen_inside']
            df.at[i, 'is_mce_inside'] = csga(con)['is_mce_inside']
            df.at[i, 'leftm'] = csga(con)['leftm']
            df.at[i, 'rightm'] = csga(con)['rightm']
            df.at[i, 'topm'] = csga(con)['topm']
            df.at[i, 'bottomm'] = csga(con)['bottomm']
        else :
            df.at[i, 'isconvex'] = np.nan
            df.at[i, 'area'] = np.nan
            df.at[i, 'aspect_ratio_wh_s'] = np.nan
            df.at[i, 'extent_s'] = np.nan
            df.at[i, 'solidity'] = np.nan
            df.at[i, 'aspect_ratio_wh'] = np.nan
            df.at[i, 'extent'] = np.nan
            df.at[i, 'ed'] = np.nan
            df.at[i, 'orien_rre'] = np.nan
            df.at[i, 'orien_ell'] = np.nan
            df.at[i, 'ratio_ell'] = np.nan
            df.at[i, 'perimeter'] = np.nan
            df.at[i, 'is_cen_inside'] = np.nan
            df.at[i, 'is_mce_inside'] = np.nan
            df.at[i, 'leftm'] = np.nan
            df.at[i, 'rightm'] = np.nan
            df.at[i, 'topm'] = np.nan
            df.at[i, 'bottomm'] = np.nan
    # Remove the rows with na
    df = df.dropna()
    # Convert true/false to 1 and 0
    df = df.replace({True: 1, False: 0})
    return df


def filter_overlap(mask):
    mb_new = mask
    mb_new1 = mask
    for i in range(0, (len(mb_new)-1)):
        for j in range((i+1), len(mb_new)):
            a = cv2.bitwise_and(mb_new[i], mb_new[j])
            al = len(np.unique(a))
            # print(i, j, "len", al)
            if al != 1:
                s1 = cv2.countNonZero(mb_new[i])
                s2 = cv2.countNonZero(mb_new[j])
                b = cv2.bitwise_or(mb_new[i], mb_new[j])
                # print(i,j, "have intersection")
                s3 = cv2.countNonZero(b)
                if s1 >= s2 :
                    # print(i,">", j)
                    if s1 == s3:
                        # print(j, "subset of", i)
                        mb_new1[j] = 0
                else:
                    # print(i,"<", j)
                    if s2 == s3:
                        # print(i, "subset of", j)
                        mb_new1[i] = 0
    emptyl = []
    for i in range(0, len(mb_new)):
        if np.all(mb_new1[i] == 0):
            emptyl.append(i)           
    mb_new1 = np.delete(mb_new1, emptyl, 0)
    # print(len(mask), len(mb_new1))
    return mb_new1


# Updated 04032024 - For entropy calculation with recognized/predicted objects/masks 
# The input dataframe should include columns 'id', 'Nr', 'p', 'count'
def entropy_calculate(df):
    total = df.groupby(['id','dir']).sum().reset_index().rename(columns = {'count':'sum'})
    total = total.drop(['p'], axis = 1)
    dft = pd.merge(df, total, on = ['id', 'dir'])
    dft['percentage'] = dft['count']/dft['sum']
    dft['id_dir'] = dft['id'] + dft['dir']
    grouped = dft.groupby('id_dir')
    grouped_arrays_list = {id_dir: dft['percentage'].iloc[indices].values for id_dir, indices in grouped.groups.items()}
    iddirl = list(grouped_arrays_list.keys())
    en = []
    for i in range(0, len(iddirl)):
        l = grouped_arrays_list[iddirl[i]]
        e = entropy(l)
        en.append(e)
    dfen = pd.DataFrame()
    dfen['id_dir'] = iddirl
    dfen['entropy'] = en
    dfentropy = pd.merge(dft, dfen, on = ['id_dir'])
    return dfentropy


# Updated 04032024 - For calculating image entropy
def entropy_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _bins = 128
    hist, _ = np.histogram(img_gray.ravel(), bins = _bins, range = (0, _bins))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist) # Default base e
    return image_entropy

# Added 09032024 - For sequential forward/backward floating feature selection - Logistic regression
# Parameters
## Direction: True or False
## Score: performance scoring metric (different for classifiers and regressors)
## cvn: number of k value for k-fold cross validation
## x: independent variables
## y: dependent variables
## return to an object
def featureselection_log(direction, score, cvn, x, y):
    sfs = SequentialFeatureSelector(linear_model.LogisticRegression(max_iter = 400),
                                k_features='best',
                                forward=direction,
                                floating=True,
                                verbose = 0,
                                scoring=score,
                                cv=cvn,
                                n_jobs = -1
                               )
    sfsr = sfs.fit(x, y)    
    return sfsr

# Added 09132024 - For sequential forward/backward floating feature selection - Linear regression
# Parameters
## Direction: True or False
## Score: performance scoring metric (different for classifiers and regressors)
## cvn: number of k value for k-fold cross validation
## x: independent variables
## y: dependent variables
## return to an object
def featureselection_lin(direction, score, cvn, x, y):
    sfs = SequentialFeatureSelector(linear_model.LinearRegression(),
                                    k_features='best',
                                    forward=direction,
                                    floating=True,
                                    verbose = 0, # For checking the logging
                                    scoring=score, 
                                    cv=cvn,
                                    n_jobs = -1 # For parallel jobs
                                   )
    sfsr = sfs.fit(x, y)    
    return sfsr

# Added 10252024 - For printing features which were recognized as significantly correlated with dependent variables
# Parameters
## r: regression result of ols summary or logit summary
def printsig(r):
    df = r.pvalues
    dfo = df[df<0.05]
    dfo = dfo.to_frame()
    dfo.loc[dfo[0] >= 0.01, 'sig'] = '*'
    dfo.loc[(dfo[0] < 0.01) & (dfo[0] >= 0.001) , 'sig'] = '**'
    dfo.loc[dfo[0] < 0.001, 'sig'] = '***'
    dfo = dfo.rename_axis('feature').reset_index()
    return dfo 

# Added 10282024 - For checking distribution of curb-related variables in y-present locations (yp = 1) and y-absent locations (yp = 1) using Kolmogorov-Smimov test
## yp is one binary dependent variable (could be accident-presence, person-injury-presence, property-damage-presence, severe-injury-presence, light-injury-presence)
## Curb-related varialbes include 'cp', 'cmean', 'cmin', 'cmax'
# Parameters
## df is a dataframe including columns of all four curb-related variables as well as yp
## dev is one target binary dependent variable yp
def check_cv_dist (df, dev):
    df1 = df.loc[df[dev] == 1]
    df0 = df.loc[df[dev] == 0]
    cvl = ['cp', 'cmean', 'cmin', 'cmax']
    print(cvl[0], ks_2samp(df1[cvl[0]], df0[cvl[0]]))
    print(cvl[1], ks_2samp(df1[cvl[1]], df0[cvl[1]]))
    print(cvl[2], ks_2samp(df1[cvl[2]], df0[cvl[2]]))
    print(cvl[3], ks_2samp(df1[cvl[3]], df0[cvl[3]]))

# Added 02202025 - For printing features which were recognized as significantly correlated with dependent variables, and their coefficients and p values
# Parameters
## r: regression result of ols summary or logit summary    
def printsigcoef(r):
    df1 = printsig(r)
    df2 = r.params.to_frame().rename_axis('feature').reset_index()
    df2 = df2.rename(columns = {0:'coef'})
    dfsigcoef = pd.merge(df1, df2, on = ['feature'])
    print(dfsigcoef)
    return dfsigcoef