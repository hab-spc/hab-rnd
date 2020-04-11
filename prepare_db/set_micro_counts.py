import os
import shutil
from datetime import datetime

import pandas as pd

from prepare_db.parse_csv import SPCParser

CELLS_ML_FLAG = True

csv_fname = '/data6/phytoplankton-db/csv/hab_micro_raw_2017_2019.csv'
if CELLS_ML_FLAG:
    csv_fname = csv_fname.replace('.csv', '_cells-mL.csv')

# Load dataset
NUM_CLASS = 9
df = pd.read_csv(csv_fname)
DATE_COL = 'SampleID'
CELL_COUNT_LIMIT_COL = 'Cell Count Detection Limit'
COUNTED_VOLUME_COL = 'Volume Counted (mL)'
DATETIME_COL = 'datetime'
CLASS_COL = 'class'
CELLS_ML_COL = 'micro cells/mL'
RAW_COUNT_COL = 'micro raw count'

spc = SPCParser()

# Preprocess classes
df = df.drop(['Polykrikos spp.', 'Prorocentrum gracile', 'Prorocentrum micans'], axis=1)
df = df.rename({
    'Akashiwo sanguinea': "Akashiwo",
    'Ceratium falcatiforme & C. fusus': "Ceratium falcatiforme or fusus",
    "Chattonella spp.": "Chattonella",
    "Cochlodinium spp.": "Cochlodinium",
    "Gyrodinium spp.": "Gyrodinium",
    'Pseudo-nitzschia spp.': 'Pseudo-nitzschia chain',
    'Prorocentrum micans + Prorocentrum spp.': 'Prorocentrum micans',
    'Total Phytoplankton (Diatoms + DinoS)': 'Total Phytoplankton'}, axis=1)
class_col = df.columns[5:-1]
print('Class columns extracted: {}'.format(class_col))

# Process dates
df[DATE_COL] = pd.to_datetime(df[DATE_COL], format='%Y%m%d').dt.strftime('%Y-%m-%d')
# df['datetime'] = df['SampleID (YYYYMMDD)'].str.cat(df['Time Collected hhmm (PST)'].astype(str), sep=' ')
dates = df[DATE_COL].to_dict()
hab_species = list(set(class_col).intersection(set(spc.hab_species[:-1])))
assert len(hab_species) == NUM_CLASS, f'Number of classes do not match {len(hab_species)} != {NUM_CLASS}'

# Get raw microscopy counts
# each cell ~ cells/Liter. Normalize this by the cell count detection limit to get the
# raw counts
temp = df[hab_species]


def stack_count_data(data, metric_name):
    data = data.stack().reset_index()
    data = data.rename({'level_0': DATETIME_COL,
                        'level_1': CLASS_COL,
                        0: metric_name}, axis=1)
    data[DATETIME_COL] = data[DATETIME_COL].map(dates)
    return data


# Get cells/mL
converted_volume_rate = 1 / 1000
temp_cells_ml = temp.iloc[:, :].multiply(1 / 1000, axis=0)
temp_cells_ml = stack_count_data(temp_cells_ml, metric_name=CELLS_ML_COL)

# Get raw count
temp = temp.iloc[:, :].div(df[CELL_COUNT_LIMIT_COL], axis=0)
temp = stack_count_data(temp, metric_name=RAW_COUNT_COL)

# Merge raw counts and cells/mL into one dataframe
temp = temp.merge(temp_cells_ml, on=[DATETIME_COL, CLASS_COL])


# Get the relative abundance and total abundance
temp['label'] = 'gtruth'
temp[RAW_COUNT_COL] = temp[RAW_COUNT_COL].round()
total_abundances = (temp.groupby(DATETIME_COL)[RAW_COUNT_COL].sum()).to_dict()
temp['micro total abundance'] = temp[DATETIME_COL].map(total_abundances)
temp['micro relative abundance'] = temp[RAW_COUNT_COL] / temp[
    'micro total abundance'] * 100.0
temp = temp.sort_values(by=[DATETIME_COL, CLASS_COL])

# Save and backup dataset
csv_fname = '/data6/phytoplankton-db/csv/hab_micro_2017_2019.csv'
if os.path.exists(csv_fname):
    backedup_csv_fname = csv_fname + f'.{datetime.now().strftime("%Y%m%d")}'
    print(f'Micro csv detected. Backing up original csv as {backedup_csv_fname}')
    shutil.copy(csv_fname, backedup_csv_fname)
temp.to_csv(csv_fname, index=False)
print(f'CSV saved as {csv_fname}')
