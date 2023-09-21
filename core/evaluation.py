import numpy as np


def compute_iou(cm):
    cm = cm.cpu().detach().numpy()
    if cm.sum() == 0:
        return 0, 0

    tp = np.diag(cm)
    with np.errstate(divide="ignore"):
        ciou = tp / (cm.sum(1) + cm.sum(0) - tp)

    miou = np.nanmean(ciou) * 100
    return ciou, miou

def compute_acc(cm):
    cm = cm.cpu().detach().numpy()
    if cm.sum() == 0:
        return 0, 0
    
    tp = np.diag(cm)
    with np.errstate(divide="ignore"):
        cacc = tp / cm.sum(1)

    macc = np.nanmean(cacc) * 100
    return cacc, macc