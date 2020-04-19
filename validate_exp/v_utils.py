""" Validation utilities file

Contains mostly computations to plot correlation graphs and other statistics
"""

# Standard Dist Imports
import logging

# Third party imports
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from scipy.stats import entropy
from sklearn.utils import resample

from transform_data import fit_distribution


def load_density_data(csv_fname, verbose=False):
    """Load the density csv data and apply transformation"""
    df = pd.read_csv(csv_fname)
    if verbose:
        print('\n{0:*^80}'.format(' Reading in the dataset '))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.info())
        print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
        print(df.head())
    return df


def retrieve_x_y(df, cls_idx=0, corrected=False, yerror=False, t='200s', smpl='',
                 micro_vs_insitu=None, micro_vs_invitro=None, invitro_vs_insitu=None):
    """Retrieve x and y values for plotting"""
    x_col_idx = 0
    y_col_idx = 1
    yerr_col_idx = 2
    columns = []

    # based on the sampling method, determine which columns to parse for
    if smpl == 'micro_vs_insitu':
        columns = [f'micro_{cls_idx}', f'pier_{cls_idx}_avg_{t}', f'pier_{cls_idx}_std_{t}']
    elif smpl == 'micro_vs_invitro':
        columns = [f'micro_{cls_idx}', f'lab_{cls_idx}_avg_{t}', f'lab_{cls_idx}_std_{t}']
    elif smpl == 'invitro_vs_insitu':
        columns = [f'lab_{cls_idx}_avg_{t}', f'lab_{cls_idx}_std_{t}',
                   f'pier_{cls_idx}_avg_{t}', f'pier_{cls_idx}_std_{t}']

    xerror_available = len(columns) > 3
    if xerror_available:
        xerr_col_idx = 1
        y_col_idx = 2
        yerr_col_idx = 3
        Xerr = df[columns[xerr_col_idx]].tolist()
    else:
        Xerr = None
        
    if yerror:
        assert 'std' in columns[yerr_col_idx]
        Yerr = df[columns[yerr_col_idx]].tolist()
    else:
        Yerr = None

    # Return as list
    X = df[columns[x_col_idx]].tolist()
    Y = df[columns[y_col_idx]].tolist()

    return X, Xerr, Y, Yerr


def best_fit(X, Y, log_scale=False, verbose=False):
    if log_scale:
        # slope, intercept, \
        # r_value, p_value, std_err = linregress(np.log10(np.array(X) + 1), np.log10(np.array(Y)+1))
        # Xfit = np.logspace(-1, 4, base=10)
        # Yfit = Xfit * slope + intercept

        x1 = [x for (x, y) in sorted(zip(X, Y))]
        y1 = [y for (x, y) in sorted(zip(X, Y))]
        x = np.array(x1)
        y = np.array(y1)
        fit = np.polyfit(x, y, deg=1)
        Xfit = x
        Yfit = fit[0] * x + fit[1]

    else:
        xbar = sum(X) / len(X)
        ybar = sum(Y) / len(Y)
        n = len(X)  # or len(Y)

        numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
        denum = sum([xi ** 2 for xi in X]) - n * xbar ** 2

        b = numer / denum
        a = ybar - b * xbar

        Yfit = [a + b * xi for xi in X]
        Xfit = X

    # Compute R2 value and other statistics from statsmodel
    res = rsquared(X, Y)

    if verbose:
        print(res.summary())
    return Xfit, Yfit

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    import statsmodels.api as sm
    X = np.hstack((np.array([1] * len(x)).reshape(-1, 1), np.array(x).reshape(-1, 1)))
    mod = sm.OLS(np.array(y).reshape(-1, 1), X)
    res = mod.fit()
    return res


def compute_values(X, Y, geometric_fit=False, log_scale=False, verbose=False):
    """Compute best fit line"""
    if geometric_fit:
        Xfit, Yfit = fit_geometric_regression(X, Y)
    else:
        Xfit, Yfit = best_fit(X, Y, log_scale=log_scale, verbose=verbose)
    return Xfit, Yfit

def fit_geometric_regression(X, Y):
    """Compute geometric regression of data"""
    from statsmodels.discrete.discrete_model import NegativeBinomial
    model = NegativeBinomial(Y, X, loglike_method='geometric')
    nbin = model.fit(disp=False)
    print(nbin.summary())


def plot_results(X, Y, Xerr=0, Yerr=0, Xfit=0, Yfit=0, label=[], class_lbl=[],
                 best_fit_line=True, log_scale=False,
                 xlabel='', ylabel='', title='', n_rows=1, n_cols=2, color='k',
                 plt_width=5, plt_height=5):
    """Wrapper function for plotting results"""
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * plt_width, n_rows * plt_height))
    mticker.Locator.MAXTICKS = 2000

    N = len(label)
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0, N, N + 1)
    norm = BoundaryNorm(bounds, cmap.N)

    print(f'Total points: {len(Y)}')
    scat = ax[0].scatter(X, Y, c=label, cmap=cmap, marker='x')
    ax[1].errorbar(X, Y, yerr=Yerr, linestyle='None', ecolor='k')
    scat1 = ax[1].scatter(X, Y, c=label, cmap=cmap, marker='x')
    cb1 = plt.colorbar(scat1, spacing='proportional', ticks=bounds)
    cb1.set_label('Classes')
    cb1.set_ticklabels(class_lbl)

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    if log_scale:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlim(1e-1, 1e4)
        ax[0].set_ylim(1e-1, 1e3)
        ax[0].set_aspect(1)

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_title(title + ' w/ errorbars')
    if log_scale:
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlim(1e-1, 1e4)
        ax[1].set_ylim(1e-1, 1e3)
        ax[1].set_aspect(1)

    if best_fit_line:
        ax[0].plot(Xfit, Yfit, color='r')
    plt.show()

def plot_dist(X):
    pass

def compute_imaged_volume(class_size=0.07):
    """Compute imaged volume given ROI size"""
    min_samp = 20
    min_pixels_per_obj = min_samp

    class_size = 1000 * np.array([class_size])

    min_resolution = class_size / min_pixels_per_obj
    pixel_size = min_resolution / 2
    blur_factor = 3
    wavelength = 0.532
    NA = 0.61 * wavelength / min_resolution
    NA[NA >= 0.75] = 0.75

    div_angle = np.arcsin(NA)
    img_DOF = blur_factor * pixel_size / np.tan(div_angle) / 2

    # compute the imaged volume in ml
    imaged_vol = pixel_size ** 2 * 4000 * 3000 * img_DOF / 10000 ** 3
    return imaged_vol

def compute_accuracies(df):
    """Compute accuracies [OUTDATED]"""
    grouped_dates_df = df.groupby('Date')
    for ii, gr in grouped_dates_df:
        false_pos_rate = (gr['False Prorocentrum'] / (gr['False Prorocentrum'] + gr['clsfier_Prorocentrum'])).values[0]
        true_pos_rate = 1.0-false_pos_rate
        false_neg_rate = (gr['False Non-Prorocentrum'] / (gr['False Non-Prorocentrum'] + gr['clsfier_Non-Prorocentrum'])).values[0]
        true_neg_rate = 1.0-false_neg_rate
        xpoint = gr['micro_proro'].values[0]
        print(f'[XAXIS {xpoint:.3f} | DATE {ii}] TRUE POS: {true_pos_rate:.3f} | FALSE POS: {false_pos_rate:.3f} || TRUE NEG: {true_neg_rate:.3f} | FALSE NEG: {false_neg_rate:.3f}')

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


def kl_divergence(p, q):
    eps = 0.01
    pp = p + eps
    pp /= sum(pp)

    qq = q + eps
    qq /= sum(qq)
    kl_div = entropy(pp, qq, base=10.)
    return kl_div


# def kl_divergence(p, q):
# return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# def smape(y_true, y_pred):
#     return 100.0 / len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true)
#                                                                        + np.abs(y_pred)))

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)


def set_counts(label, counts, micro_default=True):
    micro_counts = 'micro {}'.format('cells/mL' if micro_default else counts)
    lab_counts = f'lab {label} {counts}'
    pier_counts = f'pier {label} {counts}'
    return micro_counts, lab_counts, pier_counts


def bootstrap(x, y, data, stats, n_iterations=10000):
    """ Bootstrap dataset and measure the resulting statistic

    Usage:
        for stat in [smape, kl_divergence]:
            print(stat.__name__)
            booted_eval_metrics = {}
            settings = {'micro_vs_lab':(micro_counts, lab_counts),
                        'micro_vs_pier':(micro_counts, pier_counts),
                        'lab_vs_pier':(lab_counts, pier_counts)}
            for setting, (x, y) in settings.items():
                booted_eval_metrics[setting] = bootstrap(x=x, y=y, stats=stat, data=cls_df)
            plot_distributions(**booted_eval_metrics)

    Args:
        x:
        y:
        data:
        stats:
        n_iterations:

    Returns:

    """
    score = []
    n_size = int(len(data))
    bootstrap_size = 0
    for i in range(n_iterations):
        bootstrap_sample = resample(data, n_samples=n_size)
        score.append(stats(bootstrap_sample[x], bootstrap_sample[y]))
        bootstrap_size += len(bootstrap_sample)
    #         if i % 1000 == 0:
    #             print(f'{i}/{n_iterations} completed. Bootstrap sample size: {bootstrap_size}')

    return score


def transform_dataset(data, target_columns):
    logger = logging.getLogger(__name__)
    logger.info('Transforming dataset...')
    MICRO_ML, LAB_RC, PIER_RC, LAB_NRC, PIER_NRC = target_columns

    def _tfsm(tf_fn, tf_col, pre):
        _settings = []
        for col in tf_col:
            df[pre + col] = df[col].apply(tf_fn)

        _settings.extend(set_settings(pre + MICRO_ML, pre + LAB_RC, pre + PIER_RC))
        _settings.extend(set_settings(pre + MICRO_ML, pre + LAB_NRC, pre + PIER_NRC))

        return df, _settings

    df = data.copy()
    settings = []

    pre = 'sqrt '
    df, sqrt_settings = _tfsm(lambda x: x ** (1 / 2), target_columns, pre)

    lpre = 'logged '
    df, logged_settings = _tfsm(lambda x: np.log(x, where=0 < x), target_columns, lpre)

    lcpre = 'loggedc '
    df, loggedc_settings = _tfsm(lambda x: np.log(x + 1), target_columns, lcpre)

    # Fit distributions
    # logged_cols = {col: col.replace(' ', '_').replace('/', '_') for col in df.columns if
    #                col.startswith(lcpre)}
    # y = y.rename(logged_cols, axis=1)

    distributions = ['poisson', 'qpoisson', 'zinflate poisson', 'nbinom']

    settings_df = pd.DataFrame()
    for tgt_col in target_columns:
        fitted_data = fit_distribution(tgt_col, data, distributions)
        df = pd.concat([df, fitted_data], axis=1, sort=False)
        # get all the settings for the sample method
        settings_df[tgt_col] = list(fitted_data.columns)

    # Set up setting comparisons
    f = lambda row: (
    row[target_columns[0]], row[target_columns[-2]], row[target_columns[-1]])
    settings = settings_df.apply(f, axis=1).tolist()
    settings = [set_settings(*dist_setting) for dist_setting in settings]

    def flatten_list(_list):
        return sum(_list, [])

    settings = flatten_list(settings)

    return df, settings


def set_settings(micro, lab, pier):
    settings = [
        (micro, lab),
        (micro, pier),
        (lab, pier)
    ]
    return settings
