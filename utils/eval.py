import math
import numpy as np


def tolerance_precision(bounds, seg_bnds, tolerance_window):
    #Precision
    hit = 0.0
    for bound in seg_bnds:
        for l in range(tolerance_window + 1):
            if (bound + l in bounds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in bounds) and (bound - l > 0):
                hit += 1
                break
    return (hit / (len(seg_bnds)))


def tolerance_recall(bounds, seg_bnds, tolerance_window):
    #Recall
    hit = 0.0
    for bound in bounds:
        for l in range(tolerance_window + 1):
            if (bound + l in seg_bnds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in seg_bnds) and (bound - l > 0):
                hit += 1
                break

    return (hit / (len(bounds)))


def thresh_segmentation_eval(scores, bounds_list, tolerance_window, diff_thresh_factor):

    recall_rate_list = []
    precision_rate_list = []

    seg_bnds = []
    for i in range(len(bounds_list)):
        bnd = []
        bnd_set = set()
        bnd_set.add(0)
        diff_sorted_idx = scores[i].argsort()[::-1]

        iter_idx = 0
        #diff_thresh = diff_thresh_factor * np.fabs(np.mean(scores[i]))
        diff_thresh = diff_thresh_factor #* np.fabs(np.mean(scores[i]))
        while True:
            if iter_idx >=len(diff_sorted_idx): break

            if diff_sorted_idx[iter_idx] in bnd_set:
                iter_idx += 1
                continue

            if scores[i, diff_sorted_idx[iter_idx]] < diff_thresh:
                if len(bnd) == 0:
                    bnd.append(-1)
                break

            bnd.append(diff_sorted_idx[iter_idx])
            bnd_set.add(diff_sorted_idx[iter_idx])

            #local maximum
            bnd_set.add(diff_sorted_idx[iter_idx] + 1)
            bnd_set.add(diff_sorted_idx[iter_idx] - 1)

            iter_idx += 1

        seg_bnds.append(bnd)

    #Recall
    for i in range(len(bounds_list)):
        single_utt_recall = tolerance_recall(bounds_list[i], \
                                seg_bnds[i], tolerance_window)
        recall_rate_list.append(single_utt_recall)

    #Precision
    for i in range(len(seg_bnds)):
        single_utt_precision = tolerance_precision(bounds_list[i], \
                                seg_bnds[i], tolerance_window)
        precision_rate_list.append(single_utt_precision)

    return recall_rate_list, precision_rate_list


def r_val_eval(u_p, u_r):
    if u_r == 0 or u_p == 0:
        u_f = -1.
        u_r_val = -1.
    else:
        u_f = 2 * u_p * u_r / (u_p + u_r)
        u_os = (u_r/u_p - 1) * 100
        u_r_val = 1 - (math.fabs(math.sqrt((100-u_r)*(100-u_r) + \
         math.pow(u_os, 2))) + math.fabs( (u_r - 100 - u_os)/math.sqrt(2))) / 200
    return u_r_val * 100
