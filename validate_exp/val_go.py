import os
import sys
print(os.path.abspath(os.path.join(__file__, '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../..')))

import pandas as pd
import matplotlib.pyplot as plt

# Project level imports
from validate_exp.v_utils import plot_results, compute_values, \
    compute_accuracies,\
    load_density_data

# Module level constants
MICRO_COLUMNS = 'micro_proro'.split(',')
PRORO_COLUMNS = ['micro_proro', 'corrected_Prorocentrum',
                 'clsfier_Prorocentrum']
PHYTO_COLUMNS = 'spc_ImgCount,micro_total-phyto'.split(',')

# Initialize csv filename
data_dir = '/data6/lekevin/hab-master/hab-rnd/rawdata'
csv_fname = os.path.join(data_dir, 'Density{}_data.csv')
df = load_density_data(csv_fname.format('17-18-all'), micro_col=MICRO_COLUMNS)

time_dist = ['1min', '5min', '15min', '30min', '45min', '1h30min']
# time_dist = ['1min']

for idx, t in enumerate(time_dist):
    print('=' * 25 + f' Time Distribution: {t} ' + '=' * 25)

    # Plot layout specifications
    n_rows, n_cols = 1, 2
    plt_width, plt_height = 5, 5
    f, ax = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * plt_width, n_rows * plt_height))
    ylabel = 'Density by Clsfier predictions (cells/{})'.format(t)
    xlabel = 'Density by Microscopy (cells/mL)'
    title = 'Pred (Proro) vs Micro (Proro)'

    columns = [f'clsfier_proro_avg_{t}', f'clsfier_proro_std_{t}',
               'micro_proro']

    X, Y, Yerr, Xfit, Yfit = compute_values(columns, df, yerror=True,
                                            geometric_fit=False)

    print('total time bins per sample: {}'.format(df[f'clsfier_proro_total_smpl_{t}'].mode()[0]))
    plot_results(X, Y, Yerr, Xfit, Yfit,
                 ax=ax, idx=0, n_rows=n_rows, n_cols=n_cols,
                 xlabel=xlabel, ylabel=ylabel, title=title)

    #     zoomed_df = df
    zoomed_df = df[df.micro_proro < 15]
    X, Y, Yerr, _, _ = compute_values(columns, zoomed_df, yerror=True)
    plot_results(X=X, Y=Y, Yerr=Yerr, Xfit=Xfit, Yfit=Yfit,
                 ax=ax, idx=1, n_rows=n_rows, n_cols=n_cols,
                 xlabel=xlabel, ylabel=ylabel, title=title)
    plt.ylim(0,max(Y))
    plt.show()