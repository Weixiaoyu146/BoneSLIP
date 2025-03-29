from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

def sensitivity(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)

def specificity(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def sensitivityCalc(y_true, y_pred):
    MCM = confusion_matrix(y_true,y_pred)
    # MCM此处是 5 * 2 * 2的混淆矩阵（ndarray格式），5表示的是5分类

    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
    tn_sum = MCM[0, 0] # True Negative
    fp_sum = MCM[0, 1] # False Positive

    tp_sum = MCM[1, 1] # True Positive
    fn_sum = MCM[1, 0] # False Negative
    print("sensi tn{} fp{} tp{} fn{} ".format(tn_sum, fp_sum, tp_sum, fn_sum))
    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum + fn_sum + 1e-6

    sensitivity = tp_sum / Condition_negative
    macro_sensitivity = np.average(sensitivity, weights=None)

    micro_sensitivity = np.sum(tp_sum) / np.sum(tp_sum+fn_sum)

    return macro_sensitivity, micro_sensitivity

def specificityCalc(Labels, Predictions):
    MCM =confusion_matrix(Labels, Predictions)
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]

    tp_sum = MCM[1, 1]
    fn_sum = MCM[1, 0]
    print("speci tn{} fp{} tp{} fn{} ".format(tn_sum,fp_sum,tp_sum,fn_sum))
    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    macro_specificity = np.average(Specificity, weights=None)

    micro_specificity = np.sum(tn_sum) / np.sum(tn_sum+fp_sum)

    return macro_specificity, micro_specificity