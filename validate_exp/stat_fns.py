import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.metrics import mean_absolute_error


def mae(x, y, n=None):
    if n:
        return 1 / n * np.sum(np.abs(x - y))
    else:
        return np.mean(np.abs(x - y))


def smape(y_true, y_pred, n=None):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0

    if n:
        return 100.0 * np.sum(diff) / n
    else:
        return 100 * np.mean(diff)


def wape(y_true, y_pred):
    """Weighted Absolute Percent Error"""
    denominator = np.sum(y_true)
    diff = np.sum(np.abs(y_true - y_pred)) / denominator
    return diff


def mase(y_true, y_pred, n=None, dist=False):
    """Mean Absolute Scaled Error"""
    error = np.abs(y_true - y_pred)
    denominator = np.mean(np.abs(np.diff(y_true)))
    score = error / denominator
    if n:
        return np.sum(score) / n
    else:
        if dist:
            return np.mean(score), score
        else:
            return np.mean(score)


def investigate_mase(df, gtruth, pred):
    temp = df[['class', 'datetime', gtruth, pred]]
    temp['error'] = np.abs(temp[gtruth] - temp[pred])
    temp['naive'] = np.mean(np.abs(np.diff(temp[gtruth])))
    return temp


def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    >>> from sklearn.metrics import concordance_correlation_coefficient
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def pearson(y_true, y_pred):
    return stats.pearsonr(y_true, y_pred)

def kl_divergence(p, q):
    eps = 0.01
    pp = p + eps
    pp /= sum(pp)

    qq = q + eps
    qq /= sum(qq)
    kl_div = entropy(pp, qq, base=10.)
    return kl_div


def accuracy(gtruth, predictions):
    return np.mean(gtruth == predictions)


def error(accuracy):
    return 1.0 - accuracy


ERROR_STATS = [mean_absolute_error, smape, wape, mase]
AGREEMENT_STATS = [concordance_correlation_coefficient, kl_divergence,
                   distance.braycurtis, stats.pearsonr]
