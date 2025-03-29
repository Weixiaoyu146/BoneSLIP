import numpy as np
import math
import latexify

def calculate(true_labels, pred_labels):
    true_labels = true_labels + [1,2,3,4]
    pred_labels = np.concatenate((pred_labels, [1,2,3,4]))
    return true_labels, pred_labels

# @latexify.with_latex
# def latex():
#     # print
#     return (1_1)

if __name__ == "__main__":
    # true_labels = []
    # pred_labels = []
    # true_labels, pred_labels = calculate(true_labels, pred_labels)
    # print("true_labels", true_labels)
    # print("pred_labels", pred_labels)
    print("$math^s$")