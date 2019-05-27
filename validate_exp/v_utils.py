import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def load_density_data(csv_fname, micro_col):
    """Load the density csv data and apply transformation"""
    df = pd.read_csv(csv_fname)
    df[micro_col] = df[micro_col].apply(lambda x: x/1000, axis=1) # Liters to mL
    return df.reset_index(drop=True)

def retrieve_x_y(columns, df, corrected=False, yerror=False):
    """
    COLUMNS = ['clsifier_pred_avg', 'clsfier_pred_std', 'gtruth', 'micro']
    column indicing follows this. 
    """
    assert isinstance(columns, list)
    assert 'micro' in columns[-1]
    
    X = df[columns[-1]].tolist()
    if corrected:
        assert 'correct' in columns[-2]
        Y = df[columns[-2]].tolist()
    else:
        Y = df[columns[0]].tolist()
        
    if yerror:
        assert 'std' in columns[1]
        Yerr = df[columns[1]].tolist()
    else:
        Yerr = None
    return X, Y, Yerr

def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    Yfit = [a + b * xi for xi in X]
    Xfit = X

    r2_val = rsquared(X, Y)

    print('best fit line:\ny = {:.2f} + {:.2f}x | R^2: {}'.format(a, b, r2_val))
    return Xfit, Yfit

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def compute_values(columns, data, yerror=False, xyfit=None, corrected=False,
                   geometric_fit=False):
    X, Y, Yerr = retrieve_x_y(columns, data, corrected, yerror)
    if xyfit:
        Xfit, Yfit = xyfit
    else:
        if geometric_fit:
            Xfit, Yfit = fit_geometric_regression(X, Y)
        else:
            Xfit, Yfit = best_fit(X, Y)
    return X, Y, Yerr, Xfit, Yfit

def fit_geometric_regression(X, Y):
    """Compute geometric regression of data"""
    from statsmodels.discrete.discrete_model import NegativeBinomial
    model = NegativeBinomial(Y, X, loglike_method='geometric')
    nbin = model.fit(disp=False)
    print(nbin.summary())


def plot_results(X, Y, Yerr, Xfit, Yfit, best_fit_line=True, xlabel='',
                 ylabel='', title='', n_rows=0, n_cols=0, ax=None, idx=0, color='b'):
    if n_rows > 1:
        plt_idx = (idx // n_cols, idx%n_cols)
    else:
        plt_idx = idx

    if n_rows and n_cols:
        if Yerr:
            ax[plt_idx].errorbar(X, Y, yerr=Yerr, linestyle='None', fmt='o', ecolor=color)
        else:
            ax[plt_idx].scatter(X, Y, color=color)
        ax[plt_idx].set_xlabel(xlabel)
        ax[plt_idx].set_ylabel(ylabel)
        ax[plt_idx].set_title(title)
        if best_fit_line:
            ax[plt_idx].plot(Xfit, Yfit, color='r')
            ax[plt_idx].set_xlim(0, max(X))
    else:
        if Yerr:
            plt.errorbar(X, Y, yerr=Yerr, linestyle='None', fmt='o', ecolor=color)
        else:
            plt.scatter(X, Y, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if best_fit_line:
            plt.plot(Xfit, Yfit, color='r')
            plt.xlim(0, max(X))

def plot_dist(X):
    import matplotlib.pyplot as plt



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
    grouped_dates_df = df.groupby('Date')
    for ii, gr in grouped_dates_df:
        false_pos_rate = (gr['False Prorocentrum'] / (gr['False Prorocentrum'] + gr['clsfier_Prorocentrum'])).values[0]
        true_pos_rate = 1.0-false_pos_rate
        false_neg_rate = (gr['False Non-Prorocentrum'] / (gr['False Non-Prorocentrum'] + gr['clsfier_Non-Prorocentrum'])).values[0]
        true_neg_rate = 1.0-false_neg_rate
        xpoint = gr['micro_proro'].values[0]
        print(f'[XAXIS {xpoint:.3f} | DATE {ii}] TRUE POS: {true_pos_rate:.3f} | FALSE POS: {false_pos_rate:.3f} || TRUE NEG: {true_neg_rate:.3f} | FALSE NEG: {false_neg_rate:.3f}')
