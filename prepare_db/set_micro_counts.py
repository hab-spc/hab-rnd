from datetime import datetime
import os
import shutil

import pandas as pd
from prepare_db.parse_csv import SPCParser

csv_fname = '/data6/phytoplankton-db/csv/hab_micro_raw_summer2019.csv'
NUM_CLASS = 9
df = pd.read_csv(csv_fname)
DATE_COL = 'SampleID'
CELL_COUNT_LIMIT_COL = 'Cell Count Detection Limit'

spc = SPCParser()

# Preprocess classes
df = df.drop(['Polykrikos spp.', 'Prorocentrum gracile'], axis=1)
df = df.rename({
    'Akashiwo sanguinea': "Akashiwo",
    'Ceratium falcatiforme & C. fusus': "Ceratium falcatiforme or fusus",
    "Chattonella spp.": "Chattonella",
    "Cochlodinium spp.": "Cochlodinium",
    "Gyrodinium spp.": "Gyrodinium",
    'Pseudo-nitzschia spp.': 'Pseudo-nitzschia chain'}, axis=1)
class_col = df.columns[6:-1]
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
temp = temp.iloc[:,:].div(df[CELL_COUNT_LIMIT_COL], axis=0)

# Stack into the counts csv
temp = temp.stack().reset_index()
temp = temp.rename({'level_0': 'datetime',
                    'level_1': 'class',
                    0: 'micro raw count'}, axis=1)
temp['datetime'] = temp['datetime'].map(dates)
temp['label'] = 'gtruth'

# Get the relative abundance and total abundance
temp['micro raw count'] = temp['micro raw count'].round()
total_abundances = (temp.groupby('datetime')['micro raw count'].sum()).to_dict()
temp['micro total abundance'] = temp['datetime'].map(total_abundances)
temp['micro relative abundance'] = temp['micro raw count']/temp['micro total abundance']*100.0
temp = temp.sort_values(by=['datetime', 'class'])

csv_fname = '/data6/phytoplankton-db/csv/hab_micro_summer2019.csv'
if os.path.exists(csv_fname):
    backedup_csv_fname = csv_fname + f'.{datetime.now().strftime("%Y%m%d")}'
    print(f'Micro csv detected. Backing up original csv as {backedup_csv_fname}')
    shutil.copy(csv_fname, backedup_csv_fname)
temp.to_csv(csv_fname, index=False)
print(f'CSV saved as {csv_fname}')

# # Merge with the pier dataset
# temp = micro.merge(pier, on=['datetime', 'class'])
# temp = temp.rename({'label_y': 'label'}, axis=1)
# temp = temp.drop(['label_x'], axis=1)
# temp = temp.sort_values(by=['datetime', 'label', 'class'], axis=1)
