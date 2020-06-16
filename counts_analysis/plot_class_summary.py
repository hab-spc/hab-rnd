import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

hv.extension('bokeh')


# %%opts Scatter [tools=['hover'], width=600, height=600, legend_position='right', logx=True, logy=True, xlim=(-1, None), ylim=(-1, None)]
# %%opts Slope [logx=True, logy=True, xlim=(-1, None), ylim=(-1, None)]


def plot_class_summary(counts, data, relative=False):
    """ Plot individual summaries of each class

    Usage

    >>> plot_class_summary(rc_counts, cls_df)

    Args:
        counts:
        data:
        relative:

    Returns:

    """
    title_pre = '[Absolute Count]' if not relative else '[Relative Abundance]'
    xy = 'Count' if not relative else 'Relative Abundance'
    max_val = max(data[list(counts)].max()) + 10

    # boxwhisker plot
    rot = 0 if not relative else 5
    bx = data.groupby('datetime')[counts].sum().hvplot.box(y=list(counts),
                                                           group_label='Sampling Technique',
                                                           value_label=xy,
                                                           label='{} Distribution'.format(
                                                               title_pre),
                                                           rot=rot) \
        .opts(tools=['hover'], width=400, height=500)

    # time series
    ts = data.groupby('datetime')[counts].sum().hvplot.line(rot=30,
                                                            value_label='Total Count',
                                                            group_label='Sampling Techniques',
                                                            label=f'{title_pre} Time Series'). \
        opts(height=500, width=800, legend_position='top_right')

    # correlation plot
    dot_size, alpha = 6, 0.6

    sc1 = hv.Scatter(data, counts[0], [counts[1], 'datetime', 'class'],
                     label='lab - micro').opts(size=dot_size, alpha=alpha,
                                               tools=['hover'], )
    reg = hv.Slope.from_scatter(sc1).opts(alpha=alpha, tools=['hover'], )

    sc2 = hv.Scatter(data, counts[0], [counts[2], 'datetime', 'class'],
                     label='pier - micro').opts(size=dot_size, alpha=alpha,
                                                tools=['hover'], )
    reg2 = hv.Slope.from_scatter(sc2).opts(alpha=alpha, tools=['hover'], )

    sc3 = hv.Scatter(data, counts[1], [counts[2], 'datetime', 'class'],
                     label='pier - lab').opts(size=dot_size, alpha=alpha,
                                              tools=['hover'], )
    reg3 = hv.Slope.from_scatter(sc3).opts(alpha=alpha, tools=['hover'], )

    corr = (sc1 * sc2 * sc3 * reg * reg2 * reg3).opts(xlabel=xy, ylabel=xy,
                                                      title=f'{title_pre} Correlation',
                                                      xlim=(0, max_val),
                                                      ylim=(0, max_val), tools=['hover'],
                                                      width=500, height=500,
                                                      legend_position='right')

    cls_plot = hv.Layout(bx + ts + corr).cols(3)

    return cls_plot


def plot_summary_both_count_forms(rc_counts, rel_counts, cls_df):
    return hv.Layout(
        plot_class_summary(rc_counts, cls_df) + plot_class_summary(rel_counts, cls_df,
                                                                   relative=True)).cols(
        3).opts(shared_axes=False)


def plot_summary_sampling_class_dist(dataset, counts, logged=False, relative=False):
    """ Plots class distributions collected by each technique (hue sampling technique)

    Usage

    >>> plot_summary_sampling_class_dist(df, rc_counts, False)
    >>> plot_summary_sampling_class_dist(df, rel_counts, False, relative=True)

    Args:
        dataset:
        counts:
        logged:
        relative:

    Returns:

    """
    # === Box&Whisker ===#
    sm = dataset[['class', 'datetime'] + list(counts)]
    sm = sm.melt(id_vars=['class', 'datetime'], var_name=['setting'], value_name='count')
    sm = sm.sort_values('class')
    #     sm['setting'] = sm['setting'].map({'micro cells/mL': 'micro', 'lab gtruth raw count': 'lab', 'pier gtruth raw count':'pier'})
    plt.figure(figsize=(30, 7))
    sns.boxplot(x='class', y='count', hue='setting', data=sm)
    sns.stripplot(x='class', y='count', hue='setting', data=sm, color=".25", dodge=True)

    ylabel = 'Total Counts' if not relative else 'Relative Abundance'
    if logged:
        ylabel += ' (logged)'
        plt.yscale('symlog')
        plt.ylim(0)

    plt.ylabel(ylabel)
    plt.xlabel('Classes')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    # === Histogram ===#
    num_cols = 5
    fig, ax = plt.subplots(2, num_cols, figsize=(20, 9))
    for idx, cls in enumerate(sorted(sm['class'].unique())):
        ak_sm = sm[sm['class'] == cls]
        ak_sm_gp = ak_sm.groupby('setting')
        sns.kdeplot(ak_sm_gp.get_group(counts[0])['count'], bw=1,
                    ax=ax[idx // num_cols, idx % num_cols], label='micro')
        sns.kdeplot(ak_sm_gp.get_group(counts[1])['count'], bw=1,
                    ax=ax[idx // num_cols, idx % num_cols], label='lab')
        sns.kdeplot(ak_sm_gp.get_group(counts[2])['count'], bw=1,
                    ax=ax[idx // num_cols, idx % num_cols], label='pier')

        ax[idx // num_cols, idx % num_cols].set_title(cls + "\nDistribution")
        ax[idx // num_cols, idx % num_cols].set_xlabel(ylabel)
        ax[idx // num_cols, idx % num_cols].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
