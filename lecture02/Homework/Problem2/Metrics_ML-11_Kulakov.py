import numpy as np

def accuracy_score(y_true, y_predict, percent=None):

    if percent == None:
        pass
    else:
        y_true_sort = np.sort(y_true)
        y_true = y_true_sort[int((100 - percent)*len(y_true.reshape(-1))//100):]
        y_predict_sort = np.sort(y_predict)
        y_predict = y_predict_sort[int((100-percent)*len(y_predict.reshape(-1))//100):]
    score = np.sum([y_true == y_predict])/len(y_true.reshape(-1))
    return score

def precision_score(y_true, y_predict, percent=None):

    if percent == None:
        pass
    else:
        y_true_sort = np.sort(y_true)
        y_true = y_true_sort[int((100 - percent)*len(y_true.reshape(-1))//100):]
        y_predict_sort = np.sort(y_predict)
        y_predict = y_predict_sort[int((100-percent)*len(y_predict.reshape(-1))//100):]
    score = np.sum([y_true == 1] and [y_predict == 1]) / \
        (np.sum([y_true == 1] and [y_predict == 1]) + np.sum([y_true == 0] and [y_predict == 1]))
    return score

def recall_score(y_true, y_predict, percent=None):

    if percent == None:
        pass
    else:
        y_true_sort = np.sort(y_true)
        y_true = y_true_sort[int((100 - percent)*len(y_true.reshape(-1))//100):]
        y_predict_sort = np.sort(y_predict)
        y_predict = y_predict_sort[int((100-percent)*len(y_predict.reshape(-1))//100):]
    score = np.sum([y_true == 1] and [y_predict == 1]) / \
        (np.sum([y_true == 1] and [y_predict == 1]) + np.sum([y_true == 1] and [y_predict == 0]))
    return score

def lift_score(y_true, y_predict, percent=None):

    if percent == None:
        pass
    else:
        y_true_sort = np.sort(y_true)
        y_true = y_true_sort[int((100 - percent)*len(y_true.reshape(-1))//100):]
        y_predict_sort = np.sort(y_predict)
        y_predict = y_predict_sort[int((100-percent)*len(y_predict.reshape(-1))//100):]
    score = np.sum([y_true == 1] and [y_predict == 1]) / \
        (np.sum([y_true == 1] and [y_predict == 1]) + np.sum([y_true == 0] and [y_predict == 1])) / \
        (np.sum([y_true == 1] and [y_predict == 1]) + np.sum([y_true == 1] and [y_predict == 0]))

    return score

def f1_score(y_true, y_predict, percent=None):

    if percent == None:
        pass
    else:
        y_true_sort = np.sort(y_true)
        y_true = y_true_sort[int((100 - percent)*len(y_true.reshape(-1))//100):]
        y_predict_sort = np.sort(y_predict)
        y_predict = y_predict_sort[int((100-percent)*len(y_predict.reshape(-1))//100):]
    score = 2*precision_score(y_true, y_predict, percent) * recall_score(y_true, y_predict, percent) / \
        (precision_score(y_true, y_predict, percent) + recall_score(y_true, y_predict, percent))
    return score
