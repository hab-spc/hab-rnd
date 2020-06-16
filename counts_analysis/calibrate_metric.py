from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from counts_analysis.c_utils import COUNTS_CSV, IMG_CSV
from validate_exp.stat_fns import *

HAB_ONLY_FLAG = False
DEBUG = False
SAVE = False
scores = defaultdict(list)
datasets = ['pier', 'lab']
for cam in datasets:
    img_df = pd.read_csv(IMG_CSV[f'{cam}-pred'])
    grouped_df = img_df.groupby('image_date')


    def accuracy(gtruth, predictions):
        return np.mean(gtruth == predictions)


    def error(accuracy):
        return 1.0 - accuracy

    # read in image dataset and get all accuracies (lab and pier)
    data_counts = defaultdict(dict)
    for idx, (grp_name, grp_df) in enumerate(grouped_df):
        if HAB_ONLY_FLAG:
            grp_df = grp_df[grp_df['label'] != "Other"].reset_index(drop=True)
        gtruth, pred = grp_df['label'], grp_df['ml_hab_prediction']
        acc = accuracy(gtruth, pred)
        scores['acc'].append(acc)
        scores['error'].append(error(acc))
        scores['datetime'].append(grp_name)

    # read in counts dataset and get all MASE / WAPE
    counts_df = pd.read_csv(COUNTS_CSV['counts-v9'])
    grouped_df = counts_df.groupby('datetime')
    data_counts = defaultdict(dict)
    if DEBUG:
        for idx, (grp_name, grp_df) in enumerate(grouped_df):
            if idx <= 5:
                cls = grp_df['class']
                gtruth, pred = grp_df[f'{cam} gtruth raw count'], grp_df[
                    f'{cam} predicted raw ' \
                    f'count']

                data_counts[grp_name] = list(zip(cls, gtruth, pred))

        # Calculate by hand these values

    for grp_name, grp_df in grouped_df:
        gtruth, pred = grp_df[f'{cam} gtruth raw count'], grp_df[f'{cam} predicted raw ' \
                                                                 f'count']
        n = gtruth.sum()

        scores['mae'].append(mean_absolute_error(gtruth, pred))
        scores['wape'].append(wape(gtruth, pred))
        scores['mase'].append(mase(gtruth, pred))
        scores['smape'].append(smape(gtruth, pred))

    scores['camera'].extend([cam] * counts_df['datetime'].nunique())

scores_df = pd.DataFrame(scores)
scores_df = scores_df.sort_values('error')
if SAVE:
    filename = '/data6/lekevin/hab-master/phytoplankton-db/counts/' \
               'calibration_scores.csv'
    scores_df.to_csv(filename, index=False)

# scores_df[['error', 'mase']].plot(kind='scatter', x='error', y='mase')

from validate_exp.v_utils import best_fit

metrics_to_compare = ['mae', 'smape', 'wape', 'mase']
for y in metrics_to_compare:
    print()

    sns.scatterplot(x='error', y=y, data=scores_df, hue='camera')
    Xfit, Yfit = best_fit(scores_df['error'], scores_df[y], verbose=True)
    plt.plot(Xfit, Yfit, color='green')
    plt.title(f'Calibration Plot: Error vs {y.upper()}')
    # if y != 'smape':
    #     plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim(0, 100)
    plt.show()
