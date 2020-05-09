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
TIME_COL = 'Time Collected LOCAL PT (hhmm)'
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
df[DATETIME_COL] = df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str)
df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], format='%Y%m%d %H%M').dt.strftime(
    '%Y-%m-%d %H:%M')
# df[DATE_COL] = pd.to_datetime(df[DATE_COL], format='%Y%m%d').dt.strftime('%Y-%m-%d')
# df['datetime'] = df['SampleID (YYYYMMDD)'].str.cat(df['Time Collected hhmm (PST)'].astype(str), sep=' ')
dates = df[DATETIME_COL].to_dict()
vol_count = pd.Series(df[COUNTED_VOLUME_COL].values, index=df[DATETIME_COL]).to_dict()
cell_detection_limit = pd.Series(df[CELL_COUNT_LIMIT_COL].values,
                                 index=df[DATETIME_COL]).to_dict()
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
temp_cells_ml = temp.iloc[:, :].multiply(converted_volume_rate, axis=0)
temp_cells_ml = stack_count_data(temp_cells_ml, metric_name=CELLS_ML_COL)

# Get raw count
temp = temp.iloc[:, :].div(df[CELL_COUNT_LIMIT_COL], axis=0)
temp = stack_count_data(temp, metric_name=RAW_COUNT_COL)

# Merge raw counts and cells/mL into one dataframe
temp = temp.merge(temp_cells_ml, on=[DATETIME_COL, CLASS_COL])

# Get the relative abundance and total abundance
def compute_rel_abundance(df, count_col):
    count = count_col.replace('micro ', '')
    if count == 'raw count':
        df[count_col] = df[count_col].round()
    total_abundances = (df.groupby(DATETIME_COL)[count_col].sum()).to_dict()
    df[f'micro {count} total abundance'] = df[DATETIME_COL].map(total_abundances)
    df[f'micro {count} relative abundance'] = df[count_col] / df[
        f'micro {count} total abundance'] * 100.0
    return df

temp['label'] = 'gtruth'
temp = compute_rel_abundance(temp, RAW_COUNT_COL)
temp = compute_rel_abundance(temp, CELLS_ML_COL)
temp = temp.sort_values(by=[DATETIME_COL, CLASS_COL])
temp['micro ' + CELL_COUNT_LIMIT_COL.lower()] = temp[DATETIME_COL].map(
    cell_detection_limit)
temp['micro ' + COUNTED_VOLUME_COL.lower()] = temp[DATETIME_COL].map(vol_count)
temp[[DATETIME_COL, 'sampling time']] = temp[DATETIME_COL].str.split(' ', expand=True)

# FIX datatime
FIX_DAYLIGHT_SAVINGS_TIME_FLAG = True  # 20200408 HAB 2017_2019 has time error
if FIX_DAYLIGHT_SAVINGS_TIME_FLAG:
    print('Fixing daylight savings for valid dates')
    VALID_DATES = f'/data6/phytoplankton-db/valid_collection_dates_master.txt'
    valid_dates = open(VALID_DATES, 'r').read().splitlines()
    t = temp[temp['datetime'].isin(valid_dates)].index
    converted_times = (pd.to_datetime(temp.loc[t, 'sampling time']) + pd.DateOffset(
        hours=1)).dt.strftime("%H:%M")
    temp.loc[t, 'sampling time'] = converted_times

# Save and backup dataset
csv_fname = '/data6/phytoplankton-db/csv/hab_micro_2017_2019.csv'
if os.path.exists(csv_fname):
    backedup_csv_fname = csv_fname + f'.{datetime.now().strftime("%Y%m%d")}'
    print(f'Micro csv detected. Backing up original csv as {backedup_csv_fname}')
    shutil.copy(csv_fname, backedup_csv_fname)
temp.to_csv(csv_fname, index=False)
print(f'CSV saved as {csv_fname}')
