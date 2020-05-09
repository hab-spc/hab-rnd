from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from counts_analysis.c_utils import COUNTS_CSV, IMG_CSV
from validate_exp.stat_fns import *

pd.set_option('display.max_rows', None,
              'display.max_columns', None,
              'display.max_colwidth', -1)

HAB_ONLY_FLAG = True
DEBUG = False
scores = defaultdict(list)
datasets = ['lab', 'pier']

dates_to_check = {'lab': ['2019-08-22', '2019-08-01'],
                  'pier': ['2019-08-22', '2019-09-26']}

# grab the counts from each dataset of the dates (counts + image labels)
camera = 'lab'
grouping_count = 'datetime'
grouping_img = 'image_date'
img_df = pd.read_csv(IMG_CSV[f'{camera}-pred'])
grouped_img_dataset = img_df.groupby(grouping_img)
count_df = pd.read_csv(COUNTS_CSV['counts'])
grouped_count_dataset = count_df.groupby(grouping_count)

lb = LabelBinarizer()
lb.fit(img_df['label'])
for grp_name, grp_df in grouped_img_dataset:
    # binary_labels = lb.fit_transform(img_df['label'])
    # predicted_binary_labels = lb.transform(img_df['ml_hab_prediction'])
    binary_labels = lb.transform(grp_df['label'])
    predicted_binary_labels = lb.transform(grp_df['ml_hab_prediction'])
    print(f'Date:{grp_name}\n{"-" * 30}')
    for i in range(binary_labels.shape[1]):
        print(accuracy(binary_labels[:, i], predicted_binary_labels[:, i]))

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
