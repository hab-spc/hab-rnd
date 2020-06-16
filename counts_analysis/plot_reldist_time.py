import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Markdown, display

from counts_analysis.c_utils import get_units


def printmd(string):
    display(Markdown(string))


def plot_gtruth_average_class_distribution_over_time(count_form, y, data):
    """

    USAGE:
        # counts = list(nrmlzd_counts)
        counts = ['micro cells/mL relative abundance'] + list(rel_counts[1:])
        y = 'Average Class Percentage over N=26 Days'
        plot_gtruth_average_class_distribution_over_time(counts, y, data)

    Args:
        count_form:
        y:
        data:

    Returns:

    """
    cmap = plt.get_cmap('coolwarm')

    # Plot individual sampling techniques class distributions over each day
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for idx, smpl_technique in enumerate(count_form):
        printmd('## {}'.format(smpl_technique))
        if 'micro relative abundance' == smpl_technique:
            smpl_technique = 'micro cells/mL relative abundance'
        t = data.groupby('class')[smpl_technique].describe()
        labels = sorted(data['class'].unique())
        print(t)
        ax_idx = ax[idx]
        t['mean'].plot(kind='barh', xerr=t['std'],
                       color=cmap(np.linspace(0., 1., len(labels))), ax=ax_idx,
                       sharey=True)
        ax_idx.set_title(smpl_technique)
        ax_idx.set_xlabel(y)
        ax_idx.set_label('Class')
        if 'relative' in y:
            ax_idx.set_xlim(0, 100)
    plt.tight_layout()
    plt.show()

    # Plot sampling techniques for each class against each other
    sm = data[['class', 'datetime'] + count_form]
    sm = sm.melt(id_vars=['class', 'datetime'], var_name=['setting'], value_name=y)
    sm = sm.sort_values('class')
    plt.figure(figsize=(18, 5))
    sns.barplot(x='class', y=y, hue='setting', data=sm)
    plt.xlabel('Class')
    if 'relative' in y:
        plt.ylim(0, 100)
    plt.tight_layout()


def plot_class_distribution_per_smpl_tchnique(count_form, data, logged=True):
    """Plot class distribution for each sample technique given a count form

    prints out stats for class distribution that make up overall distribution

    """
    for smpl_technique in count_form:
        data_dict = data.groupby('class')[smpl_technique]
        printmd('### {}'.format(smpl_technique))
        print('Class Distribution collected over N=26 Days')
        print(data.groupby('class')[smpl_technique].describe())
        printmd('#### Overall Distribution')
        printmd('#### Left to Right: Class Histogram | Class Distribution')
        _plot_dataset_distribution(data_dict.sum().to_dict())

        printmd('#### Time Series')
        _plot_time_series(smpl_technique, data, logged=logged)


def _plot_dataset_distribution(data_dict):
    """Plot dataset distribution in the form of raw counts and normalized counts

    USAGE:
        >> _plot_dataset_distribution({'Akashiwo': 1, 'Ceratium furca': 25}, colors)
    """
    # Grab data from dictionary
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())
    num_samples = range(len(labels))

    # Assign color
    cmap = plt.get_cmap('coolwarm')

    # Generate plot
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    bars = plt.bar(num_samples, sizes, color=cmap(np.linspace(0., 1., len(labels))))
    plt.xlabel('Classes')
    plt.ylabel('Total Counts')
    plt.xticks(num_samples, labels, rotation=45, fontsize=12)

    # Assign text to top of bars
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%d' % int(height),
                 ha='center', va='bottom', fontsize=14)

    # Plot Pie chart of the distribution
    plt.subplot(1, 2, 2)
    plt.pie(sizes, labels=labels, colors=cmap(np.linspace(0., 1., len(labels))),
            autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize':
                                                                           14})
    plt.axis('equal')
    plt.tight_layout()

    plt.show()


def _plot_time_series(smpl_technique, data, logged=True):
    """
    USAGE:
        _plot_time_series('lab gtruth raw count', data, logged=True)
    """
    current_palette_7 = sns.color_palette("coolwarm", 9)
    sns.set_palette(current_palette_7)

    plt.figure(figsize=(18, 5))
    data = data.sort_values('class')
    sns.lineplot(x='datetime', hue='class', y=smpl_technique, data=data)
    log_str = ''
    if logged:
        log_str = 'Logged '
        plt.yscale('symlog')

    plt.ylabel('{}Count ({})'.format(log_str, get_units(smpl_technique)))
    plt.ylim(0, 1100)
    plt.xticks(rotation=45)
    plt.xlabel('Date Sampled')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
