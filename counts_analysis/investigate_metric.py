import warnings
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from counts_analysis.c_utils import COUNTS_CSV, IMG_CSV
from validate_exp.stat_fns import *

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None,
              'display.max_columns', None,
              'display.max_colwidth', -1)

HAB_ONLY_FLAG = True
DEBUG = False
INVESTIGATE_SMAPE = False
INVESTIGATE_MASE = False

scores = defaultdict(list)
datasets = ['pier', 'lab']

# This is all done for the classifier (lab, pier) then the cameras
# Investigate error metrics for 3 classes (Akashiwo, Gyrodinium, Prorocentrum micans)
CLASSES = ['Akashiwo', 'Gyrodinium', 'Prorocentrum micans']
# Load dataset
count_df = pd.read_csv(COUNTS_CSV['counts-v9'])

grouping = 'datetime'
grouped_df = count_df.groupby(grouping)

if INVESTIGATE_SMAPE or INVESTIGATE_MASE:
    if INVESTIGATE_MASE:
        date = '2019-06-24'
        smpl_data = grouped_df.get_group(date)
        camera = 'pier'
        x, y = f'{camera} gtruth raw count', f'{camera} predicted raw count'
        print(smpl_data[[x, y]])
        print(abs(np.diff(smpl_data[x])))

        smpl_data['error'] = np.abs(smpl_data[x] - smpl_data[y])
        smpl_data['naive'] = np.mean(abs(np.diff(smpl_data[x])))
        smpl_data['score'] = smpl_data['error'] / smpl_data['naive']
        print(smpl_data['score'].mean())

grouping = 'class'
grouped_df = count_df.groupby(grouping)
class_counts = defaultdict(dict)
scores = defaultdict(list)  # class: (lab, score), (pier, score)

if INVESTIGATE_SMAPE or INVESTIGATE_MASE:
    from eval_counts import evaluate_counts

    cls = 'Akashiwo'
    camera = 'pier'
    x, y = f'{camera} gtruth raw count', f'{camera} predicted raw count'
    stat = smape
    smpl_data = grouped_df.get_group(cls)
    score = evaluate_counts(smpl_data, gtruth=x, pred=y, stat=stat)
    print(score)

    if INVESTIGATE_SMAPE:
        # Build up the smape
        smpl_data['error'] = np.abs(smpl_data[x] - smpl_data[y])
        smpl_data['sum'] = smpl_data[x] + smpl_data[y]
        smpl_data['percentage'] = smpl_data['error'] / smpl_data['sum']
        smpl_data['percentage'] = smpl_data['percentage'].fillna(0)
        print(smpl_data['percentage'].mean())
        # SMAPE > total observations of 5
        smape_data = smpl_data[smpl_data['percentage'] > 5]
        print(smpl_data['percentage'].mean())

    if INVESTIGATE_MASE:
        smpl_data['error'] = np.abs(smpl_data[x] - smpl_data[y])
        smpl_data['naive'] = np.mean(abs(np.diff(smpl_data[x])))
        smpl_data['score'] = smpl_data['error'] / smpl_data['naive']
        print(smpl_data['score'].mean())

for camera in datasets:
    for cls in CLASSES:

        x_col, y_col = f'{camera} gtruth raw count', f'{camera} predicted raw count'

        # Get data
        data = grouped_df.get_group(cls)[[grouping, 'datetime', x_col, y_col]]
        data['error'] = np.abs(data[x_col] - data[y_col])
        class_counts[camera][cls] = data
        print(f'Camera: {camera} Class: {cls}')

        samples = list(range(2, len(data) + 1, 2))

        stats = ERROR_STATS
        for smpl_size in samples:
            data_ = data[:smpl_size]

            # Investigate counts
            scores['number of dates'].append(smpl_size)
            scores['class'].append(cls)
            scores['camera'].append(camera)
            for idx, stat in enumerate(stats):
                x, y = data_[x_col], data_[y_col]
                # if stat.__name__ == 'mean_absolute_error':
                #     data_['error'] = np.abs(x - y)
                #
                # elif stat.__name__ == 'smape':
                #     data_['error'] = np.abs(x - y)
                #     data_['total_observations'] = x + y

                score = stat(x, y)
                scores[stat.__name__].append(score)

scores_df = pd.DataFrame(scores)

# Investigate error metrics for 2 dates


dates_to_check = {'lab': ['2019-08-22', '2019-08-01'],
                  'pier': ['2019-08-22', '2019-09-26']}

# grab the counts from each dataset of the dates (counts + image labels)
camera = 'pier'
grouping_count = 'datetime'
grouping_img = 'image_date'

img_df = pd.read_csv(IMG_CSV[f'{camera}-pred'])
grouped_img_dataset = img_df.groupby(grouping_img)
count_df = pd.read_csv(COUNTS_CSV['counts'])
grouped_count_dataset = count_df.groupby(grouping_count)

lb = LabelBinarizer()
lb.fit(img_df['label'])
# average accuracy
scores = np.zeros((26, 10))
for idx, (grp_name, grp_df) in enumerate(grouped_img_dataset):
    # binary_labels = lb.fit_transform(img_df['label'])
    # predicted_binary_labels = lb.transform(img_df['ml_hab_prediction'])
    binary_labels = lb.transform(grp_df['label'])
    predicted_binary_labels = lb.transform(grp_df['ml_hab_prediction'])
    print(f'Date:{grp_name}\n{"-" * 30}')
    for i in range(binary_labels.shape[1]):
        acc = accuracy(binary_labels[:, i], predicted_binary_labels[:, i])
        scores[idx, i] = acc

        print(i, acc)

print(np.mean(scores, axis=0))

gtruth = lb.transform(img_df['label'])
predictions = lb.transform(img_df['ml_hab_prediction'])

class_scores = []
for i in range(binary_labels.shape[1]):
    acc = accuracy(gtruth[:, i], predictions[:, i])
    class_scores.append(acc)

print(class_scores)

date = dates_to_check[camera][0]

cx, cy = f'{camera} gtruth raw count', f'{camera} predicted raw count'
ix, iy = 'label', 'ml_hab_prediction'

cdata = grouped_count_dataset.get_group(date)[['class', cx, cy]]
cdata = cdata.sort_values(cx)
idata = grouped_img_dataset.get_group(date)[[ix, iy]]

# Print out distribution for counts and image datasets
print(cdata)

# Evaluate metrics
score = 0
acc = accuracy(idata[ix], idata[iy])
err = mase(cdata[cx], cdata[cy])

df = cdata.copy()
df['error'] = np.abs(df[cx] - df[cy])
df['naive'] = np.mean(np.abs(np.diff(df[cx])))
