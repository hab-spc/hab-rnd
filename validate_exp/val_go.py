"""Module for plotting the sampling correlations"""

# Standard dist
import os
import sys
print(os.path.abspath(os.path.join(__file__, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

# Third party imports
import pandas as pd

# Project level imports
from validate_exp.v_utils import plot_results, compute_values, retrieve_x_y

# Module level constants
# HAB species of interest
CLASSES = {'Akashiwo': 1,
           'Ceratium falcatiforme or fusus': 2,
           'Ceratium furca': 3,
           'Ceratium other': 4,
           'Chaetoceros socialis': 5,
           'Chattonella': 6,
           'Ciliates': 7,
           'Cochlodinium': 8,
           'Dinophysis': 9,
           'Eucampia': 10,
           'Gyrodinium': 11,
           'Lingulodinium polyedra': 12,
           'Polykrikos': 14,
           'Prorocentrum micans': 15,
           'Prorocentrum spp': 16,
           'Pseudo-nitzschia chain': 17}


def plot_sampling_correlation(df, smpl='micro_vs_insitu', plot_classes=False, compute_log_scale=True, t='200s',
                              verbose=False):
    """ Plot sampling correlation between two sampling methods

    Accepts dataframe holding columns of the sampling counts w.r.t to a given class (i.e. micro_0, pier_0_avg_200s),
    then plotting it on a scatter plot for all classes.

    Args:
        df (pd.DataFrame): Density dataframe holding sampling counts
        smpl (str): Sampling methods to compare [options: 'micro_vs_insitu', 'micro_vs_invitro', 'invitro_vs_insitu']
        plot_classes (bool): Flag to view individual class plots prior to all classes
        compute_log_scale (bool): Plot the correlation graph on both x and y log axes
        t (str): Time frequency
        verbose (bool): Verbosity flag

    Returns:
        None

    """
    # Initialize plot parameters
    n_rows, n_cols = 1, 2
    plt_width, plt_height = 5, 5
    x, y = smpl.split('_')[0], smpl.split('_')[2]
    ylabel = 'Avg density from {} data (cells/{})'.format(y, t)
    xlabel = 'Density from Micro data (cells/mL)' if x == 'micro' else 'Avg density from In Vitro data (cells/{})'.format(
        t)
    title = smpl

    # Initialize to store results
    Xresults, Yresults, Yerr_results, label = [], [], [], []

    # For each class, retrieve the sampling counts for both the x and y axis and/or plot the class correlation
    for cls_name, cls_idx in CLASSES.items():
        # Parse through the dataframe for the x and y coordinates given the `smpl`
        X, _, Y, Yerr = retrieve_x_y(df, cls_idx, yerror=True, smpl=smpl)

        if plot_classes:
            # Compute the best fit line and plot
            Xfit, Yfit = compute_values(X, Y, log_scale=compute_log_scale)
            plot_results(X, Y, Yerr=Yerr, Xfit=Xfit, Yfit=Yfit, label=[cls_idx] * len(Y),
                         class_lbl=list(CLASSES.keys()),
                         n_rows=n_rows, n_cols=n_cols, log_scale=True,
                         xlabel=xlabel, ylabel=ylabel, title=title + '(Class {} | {})'.format(cls_idx, cls_name))

        # Store class results to put all into one plot
        Xresults += X
        Yresults += Y
        Yerr_results += Yerr
        label += [cls_idx] * len(Y)

    # Recompute new best fit line for all classes case and plot
    Xfit, Yfit = compute_values(Xresults, Yresults, log_scale=compute_log_scale, verbose=verbose)
    plot_results(Xresults, Yresults, Yerr=Yerr_results, Xfit=Xfit, Yfit=Yfit, label=label,
                 class_lbl=list(CLASSES.values()),
                 n_rows=n_rows, n_cols=n_cols, log_scale=True,
                 xlabel=xlabel, ylabel=ylabel, title=title + ' (All Classes)')


if __name__ == '__main__':
    pass
