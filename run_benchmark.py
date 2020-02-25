"""

Get data (microscopy, in situ, lab)
Create density csv file
val_go.py (plot_results() )

"""
# Standard dist imports
import datetime
import logging
import os
from pprint import pformat

import subprocess

# Third party imports
import pandas as pd

# Project level imports
from prepare_db.create_csv import create_density_csv, create_time_period_csv
import importlib
importlib.import_module('hab-ml')
from main import deploy
from spc.spcserver import SPCServer
from utils.config import set_config
from utils.logger import Logger

# Module level constants
DATA_DIR = 'rawdata'
VERSION = 'Prorocentrum_20190523-0610'
MICRO_CSV = os.path.join(DATA_DIR, 'Micro-{}.csv'.format(VERSION))
INSITU_CSV = os.path.join(DATA_DIR, 'SPC-{}.csv'.format(VERSION))
LAB_CSV = os.path.join(DATA_DIR, 'LAB-{}.csv'.format(VERSION))

def get_predicted_insitu_images():
    """

    One-time use of time period txt files. Overwrites it all of the time

    Returns:

    """
    # Initialize Logger
    root_dir = 'experiments/hab_fieldv2'
    Logger(log_filename=os.path.join(root_dir, 'insitu-{}.log'))
    logger = logging.getLogger('insitu-deployment')
    Logger.section_break('Insitu Deployment')

    # Create time period text for SPICI
    time_periods = os.path.join(root_dir, 'pilot_benchmark.txt')
    create_time_period_csv(output_csv=time_periods, micro_csv=MICRO_CSV, timefmt='%H:%M')
    print('Time period txt file created.')

    # Deploy SPICI
    hab_db = '/data6/phytoplankton-db/'
    image_dir = os.path.join(hab_db, 'hab_in_situ/hab_field/field_20190523-0610')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    spc = SPCServer().retrieve(textfile=time_periods,
                               data_dir=image_dir,
                               output_csv_filename=INSITU_CSV,
                               download=True)

    # Deploy classifier
    opt = set_config(deploy_data=INSITU_CSV,
                     batch_size=256,
                     mode='deploy')
    logger.info(pformat(opt._state_dict()))
    deploy(opt, logger)


def get_predicted_invitro_images(save=False):
    # Get common dates between Microscopy csv and available lab images
    hab_db = '/data6/phytoplankton-db/'
    df = pd.read_csv(MICRO_CSV)
    dates = pd.to_datetime(df['Datemm/dd/yy']).dt.strftime('%Y%m%d').to_list()
    lab_df = pd.DataFrame()
    image_dir = os.path.join(hab_db, 'hab_in_vitro/images')
    for date in dates:
        rel_path = '001/00000_static_html'
        pred_json = os.path.join(image_dir, date, rel_path, 'gtruth.json')
        csv_fname = os.path.join(image_dir, date, rel_path, 'features.csv')

        # join predictions into features.csv

        # concat into LAB CSV
        try:
            lab_df = lab_df.append(pd.read_csv(csv_fname))
        except FileNotFoundError:
            print('File does not exist: {}'.format(csv_fname))
            continue

    if save:
        lab_df.to_csv(LAB_CSV, index=False)


def main():
    # Get available dates and sample times
    assert os.path.exists(MICRO_CSV), 'Ensure microscopy csv exists'

    # Get in situ data
    get_predicted_insitu_images()

    # Get in vitro data
    get_predicted_invitro_images()

    # Create density csv file
    create_density_csv(output_dir=DATA_DIR,
                       micro_csv=MICRO_CSV,
                       insitu_csv=INSITU_CSV,
                       log_fname='Density-{}.log'.format(VERSION),
                       csv_fname='Density-{}.csv'.format(VERSION))




if __name__ == '__main__':
    main()