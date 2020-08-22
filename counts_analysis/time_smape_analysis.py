import matplotlib.pyplot as plt
import seaborn as sns

ERROR = 'smape'
TIME = 'time'

def set_labels_axes(ax_idx=None, logged=True):
    x_label = 'Time Window(sec)'
    if logged:
        x_label = 'Logged ' + x_label
    y_label = ERROR.upper()
    if ax_idx:
        # X Axis
        if logged:
            ax_idx.set_xscale('log')
        ax_idx.set_xlabel(x_label)
        # Y Axis
        ax_idx.set_ylabel(y_label)
    else:
        if logged:
            plt.xscale('log')
        plt.xlabel(x_label)
        plt.ylabel(y_label)


def plot_smape_vs_time_allsettings(lab_micro, pier_micro, pier_lab, logged=True):
    # Micro vs Pier
    plot_smape_vs_time(lab_micro, color='blue', label='lab - micro')

    # Micro vs Pier
    plot_smape_vs_time(pier_micro, color='orange', label='pier - micro')

    # Lab vs Pier
    plot_smape_vs_time(pier_lab, color='green', label='pier - lab')

    # Combined Settings 
    plt.figure()
    sns.lineplot(pier_micro[TIME], pier_micro[ERROR], color='orange', label='pier - micro')
    sns.lineplot(pier_lab[TIME], pier_lab[ERROR], color='green', label='pier - lab')
    set_labels_axes(logged=logged)
    plt.legend()
    plt.show()
    
def plot_smape_vs_time(data_dict, label='', color='blue', logged=True):
    """
    
    Args:
        data_dict: Key: ERROR, TIME Values: list of scores & time values
        logged: Flag for logged scale 

    Returns:

    """
    if logged:
        fig, ax = plt.subplots(1,2, figsize=(15, 5))
        ax_idx = ax[0]
        sns.lineplot(data_dict[TIME], data_dict[ERROR], ax=ax_idx, label=label,
                     color=color)
        set_labels_axes(ax_idx, logged)
        ax_idx = ax[1]
        sns.lineplot(data_dict[TIME], data_dict[ERROR], ax=ax_idx, label=label,
                     color=color)
        set_labels_axes(ax_idx, logged)
        plt.show()
    else:
        sns.lineplot(data_dict[TIME], data_dict[ERROR], label=label, color=color)
        set_labels_axes(logged=logged)
        plt.show()

