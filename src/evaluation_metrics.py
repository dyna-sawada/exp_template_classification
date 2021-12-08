


import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import label_ranking_loss, coverage_error



def f1_threshold_score_label(y_trues: np.array, y_preds: np.array, threshold=0.5, average='macro', zero_division=0):

    assert y_trues.shape == y_preds.shape, "Numpy array's shape is not the same between preds and trues."

    y_preds_01 = np.where(y_preds >= threshold, 1, 0)

    f1 = f1_score(y_preds_01, y_trues, average=average, zero_division=zero_division)

    return f1



def pr_auc_scores_label(y_trues: np.array, y_preds: np.array, average='macro'):
    
    assert y_trues.shape == y_preds.shape, "Numpy array's shape is not the same between preds and trues."
    
    flag_compute = np.sum(y_trues, axis=0) > 0

    y_preds = y_preds[:, flag_compute]
    y_trues = y_trues[:, flag_compute]

    pr_scores = average_precision_score(y_trues, y_preds, average=None)
    pr_average = average_precision_score(y_trues, y_preds, average=average)

    for idx, flag in enumerate(flag_compute):
        if flag == False:
            pr_scores = np.insert(pr_scores, idx, np.nan)

    ## = number of template ids
    assert len(pr_scores) == y_trues.shape[1] + len([flg for flg in flag_compute if flg == False])

    return pr_scores, pr_average



def roc_auc_scores_label(y_trues: np.array, y_preds: np.array, average='macro'):

    assert y_trues.shape == y_preds.shape, "Numpy array's shape is not the same between preds and trues."
    
    flag_compute = np.sum(y_trues, axis=0) > 0

    y_preds = y_preds[:, flag_compute]
    y_trues = y_trues[:, flag_compute]

    roc_scores = roc_auc_score(y_trues, y_preds, average=None)
    roc_average = roc_auc_score(y_trues, y_preds, average=average)

    for idx, flag in enumerate(flag_compute):
        if flag == False:
            roc_scores = np.insert(roc_scores, idx, np.nan)

    ## = number of template ids
    assert len(roc_scores) == y_trues.shape[1] + len([flg for flg in flag_compute if flg == False])

    return roc_scores, roc_average



def one_error_score_label(y_trues: np.array, y_preds: np.array):

    assert y_trues.shape == y_preds.shape, "Numpy array's shape is not the same between preds and trues."

    y_preds_t = y_preds.T
    y_trues_t = y_trues.T

    one_error_list = []
    for y_p, y_t in zip(y_preds_t, y_trues_t):
        sorted_idx = np.argsort(-y_p)
        y_p = y_p[sorted_idx]
        y_t = y_t[sorted_idx]

        if np.sum(y_t) == 0:
            score = np.nan
        else:
            count = 0
            for _p, t in zip(y_p, y_t):
                if t == 1:
                    break
                else:
                    count += 1
            score = count / len(y_t)

        one_error_list.append(score)

    assert len(one_error_list) == 25

    one_error_scores = np.array(one_error_list)
    one_error_average = np.nanmean(one_error_scores)
    
    return one_error_scores, one_error_average



def _one_error(y_true, y_pred):
    count = 0
    for y_p, y_t in zip(y_pred, y_true):
        top_cls = np.argmax(y_p)
        if y_t[top_cls] != 1:
            count += 1

    return count / len(y_pred)



def coverage_score_label(y_trues: np.array, y_preds: np.array):

    assert y_trues.shape == y_preds.shape, "Numpy array's shape is not the same between preds and trues."

    y_preds_t = y_preds.T
    y_trues_t = y_trues.T

    cov_score = coverage_error(y_trues_t, y_preds_t)

    return cov_score



def ranking_loss_score_label(y_trues: np.array, y_preds: np.array):

    assert y_trues.shape == y_preds.shape, "Numpy array's shape is not the same between preds and trues."

    y_preds_t = y_preds.T
    y_trues_t = y_trues.T

    rank_loss_score = label_ranking_loss(y_trues_t, y_preds_t)

    return rank_loss_score


