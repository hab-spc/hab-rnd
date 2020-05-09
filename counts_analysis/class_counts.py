import matplotlib.pyplot as plt
import seaborn as sns

from counts_analysis.c_utils import get_units
from validate_exp.v_utils import best_fit


def set_labels_axes(ax_idx=None, logged=True, x_label=None, y_label=None, title=None):
    if logged:
        x_label = 'Logged ' + x_label
        y_label = 'Logged ' + y_label
    if ax_idx:
        # X Axis
        if logged:
            ax_idx.set_xscale('symlog')
            ax_idx.set_xlim(0)
        ax_idx.set_xlabel(x_label)
        # Y Axis
        if logged:
            ax_idx.set_yscale('symlog')
            ax_idx.set_ylim(0)
        ax_idx.set_ylabel(y_label)
        # Title
        ax_idx.set_title(title)
    else:
        if logged:
            plt.xscale('symlog')
            plt.yscale('symlog')
            plt.ylim(0)
            plt.xlim(0)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)


def plot_distribution_all_settings(data, counts, ax=None):
    xlabel = 'Collected Sample Size'
    title = 'Collected Sample Size Distribution'
    data[['datetime'] + list(counts)].plot(kind='hist', x='datetime',
                                           figsize=(18, 5), ax=ax, alpha=0.4)
    ax.set_xlabel(xlabel) if ax else plt.xlabel(xlabel)
    ax.set_title(title) if ax else plt.title(title)


def plot_time_series(data, counts, ylabel='', title='', ax=None):
    data[['datetime'] + list(counts)].plot(kind='line', x='datetime',
                                           figsize=(18, 5), ax=ax)
    ax.set_ylabel(ylabel) if ax else plt.ylabel(ylabel)
    ax.set_title(title) if ax else plt.title(title)


def plot_scatterplot_all_settings(settings, data, logged=False, verbose=False):
    NUM_COLS = 3
    fig, ax = plt.subplots(1, NUM_COLS, figsize=(15, 5))
    for i_ax, setting in enumerate(settings):
        ax_idx = ax[i_ax]

        x, y = settings[setting]
        sns.scatterplot(x=data[x], y=data[y], ax=ax_idx)
        set_labels_axes(ax_idx=ax_idx, logged=logged,
                        x_label='{} Counts ({})'.format(
                            'Manual' if 'lab' not in x else 'Camera', get_units(x)),
                        y_label='Camera Counts ({})'.format(get_units(y)), title=setting)
        Xfit, Yfit = best_fit(data[x], data[y], logged, verbose=verbose)
        ax_idx.plot(Xfit, Yfit, color='orange')
    plt.tight_layout()
    plt.show()


def plot_scatterplot_gtruth_pred(data, gtruth_pred_counts, ax=None, title='',
                                 logged=False, verbose=False):
    x, y = gtruth_pred_counts[0], gtruth_pred_counts[1]
    data[gtruth_pred_counts].plot(kind='scatter', x=x, y=y, ax=ax)
    Xfit, Yfit = best_fit(data[x], data[y], logged, verbose=verbose)
    ax.plot(Xfit, Yfit, color='blue') if ax else plt.plot(Xfit, Yfit, color='blue')
    ax.set_title(title) if ax else plt.title(title)
