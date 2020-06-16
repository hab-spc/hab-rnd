import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Markdown, display


def printmd(string):
    display(Markdown(string))


def plot_summary_daily_counts(date_data, rc_counts, rel_counts):
    """

    Usage

    >>> for date, date_data in data.groupby('datetime'):
    >>>     date_data1 = date_data.sort_values(by='class')
    >>>     plot_summary_daily_counts(date_data1)

    Args:
        date_data:

    Returns:

    """
    date = date_data['datetime'].unique()[0]
    printmd(f"# ======= {date} ======= #")
    printmd(f"### Summary Statistics for {date}")

    count = 'lab gtruth raw count'
    class_lbls = date_data['class'].sort_values()
    class_counts = date_data.sort_values(by='class')[count]

    display(date_data[list(rc_counts)].describe())

    printmd(f"### Class Counts for each Sampling Technique")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 3, 1)
    micro, lab, pier = rc_counts
    plt.title(f'{micro} (N={date_data[micro].sum()})')
    plot_bar(data_dict=dict(zip(date_data['class'], date_data[micro])))
    ax2 = fig.add_subplot(1, 3, 2)
    plt.title(f'{lab} (N={date_data[lab].sum()})')
    plot_bar(data_dict=dict(zip(date_data['class'], date_data[lab])), sharey=True)
    ax3 = fig.add_subplot(1, 3, 3)
    plt.title(f'{pier} (N={date_data[pier].sum()})')
    plot_bar(data_dict=dict(zip(date_data['class'], date_data[pier])), sharey=True)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.show()

    printmd("### Raw Data (Absl & Relative Counts)")
    counts = sorted(list(rc_counts) + list(rel_counts))
    display(date_data[['class', 'datetime'] + counts])

    printmd(
        f"### Class Percentages for each Sampling Technique (w.r.t total species count of the period)")
    y = 'Relative Abundance'
    sm = date_data[['class', 'datetime'] + list(rel_counts)]
    sm = sm.melt(id_vars=['class', 'datetime'], var_name=['setting'], value_name=y)
    plt.figure(figsize=(28, 8))
    ax = sns.barplot(x='class', y=y, hue='setting', data=sm)
    plt.xlabel('Class')
    # plt.xticks(rotation=30)
    plt.ylim(0, 100)
    plt.ylabel('Relative Abundance')
    plt.yticks(fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    for p in ax.patches:
        ax.annotate("{:0.2f}".format(p.get_height()), (p.get_x(), p.get_height() * 1.15),
                    fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_bar(data_dict, ax_idx=None, sharey=False):
    # Grab data from dictionary
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())
    num_samples = range(len(labels))

    # Assign color
    cmap = plt.get_cmap('coolwarm')

    # Generate plot
    bars = plt.barh(num_samples, sizes, color=cmap(np.linspace(0., 1., len(labels))))
    plt.xlabel('Total Counts')

    if not sharey:
        plt.ylabel('Classes')
        plt.yticks(num_samples, labels, fontsize=12)

    # Assign text to top of bars
    fmt = '.2f'
    for rect, size in zip(bars, sizes):
        width = rect.get_width()
        plt.text(width + 10, rect.get_y() + rect.get_height() / 2, format(size, fmt),
                 ha='center', va='bottom')
