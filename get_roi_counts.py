import os

# Third party imports
import pandas as pd

from prepare_db.create_csv import create_time_period_csv
from prepare_db.parse_csv import SPCParser

"""UPDATE on local raw_data/hab_in_situ_summer19_times.csv then copy it to the 
parent directory of the micro csv"""

#=== Create meta file ===#
output_csv_summer19 = '/data6/lekevin/hab-master/spici/DB/meta/summer2019/time_period.txt'
micro_csv_summer19 = '/data6/lekevin/hab-master/hab_rnd/experiments/exp_hab20_summer2019/hab_in_situ/hab_in_situ_summer19_times.csv'
create_time_period_csv(output_csv_summer19, micro_csv_summer19, timefmt='%H:%M',
                       offset_min=17, min_camera=0.03,
                       max_camera=1.0, date_col='SampleID (YYYYMMDD)',
                       time_col='Time Collected hhmm (PST)')

"""HAVE TO DO THIS THROUGH TERMINAL"""
#=== Pull the data using spici ===#
pull_images_cmd = """python spc_go.py --search-param-file=DB/meta/summer2019/time_period.txt --image-output-path=DB/images/summer2019 --meta-output-path=DB/csv/hab_in_situ_summer2019_1.csv -d"""
os.system(pull_images_cmd)

pull_images_cmd = """python spc_go.py --search-param-file=DB/meta/selfsupervision/time_period.txt --image-output-path=DB/images/selfsupervision --meta-output-path=DB/csv/hab_in_situ_selfsupervision.csv -d"""


#=== Create Dataframe for getting Roi counts)
csv_fname = '/data6/phytoplankton-db/csv/hab_in_situ_summer2019_1.csv'
df = pd.read_csv(csv_fname)
time_df = pd.read_csv('/data6/lekevin/hab-master/spici/DB/meta/summer2019/time_period.txt', sep=',',
                        names=['start_time', 'end_time', 'min_len', 'max_len', 'cam'])
time_df['image_date'] = pd.to_datetime(time_df['start_time']).dt.date.astype(str)
temp = df.merge(time_df, on='image_date')


#=== use SPC RoI count function ===#
spc = SPCParser(csv_fname)
data = spc.get_ROI_counts(temp)
for k,v in data.items():
    print(f'{v["start_time"]},{v["end_time"]},{v["ROI_count"]}')

for k, v in ROI_counts.items():
    print('\n{0:*^80}'.format(' {} ({}) '.format(k, v["ROI_count"])))
    print(f'Date: {k}')
    print(f'Total ROIs: {v["ROI_count"]}')
    print(f'HAB Species Distribution\n{"-" * 30}')
    for cls, count in v["gtruth_dist"].items():
        print('{:50} {:5}'.format(cls, count))
