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
MICRO_CSV = os.path.join(DATA_DIR, 'Micro-{}.csv'.format('Prorocentrum_20190523-20190610'))
INSITU_CSV = os.path.join(DATA_DIR, 'SPC-{}.csv'.format('Prorocentrum_20190523-20190610'))
LAB_CSV = os.path.join(DATA_DIR, 'LAB-all_data.csv')

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
    logger.info('Time period txt file created.')

    # Deploy SPICI
    hab_db = '/data6/phytoplankton-db/'
    image_dir = os.path.join(hab_db, 'hab_in_situ/hab_field/field_20190523-0610')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    spc = SPCServer().retrieve(textfile=time_periods,
                             output_dir=image_dir,
                             output_csv_filename=INSITU_CSV,
                             download=True)

    # Deploy classifier
    opt = set_config(deploy_data=INSITU_CSV,
                     batch_size=256,
                     mode='deploy')
    logger.info(pformat(opt._state_dict()))
    deploy(opt, logger)


def main():
    # Get available dates and sample times
    assert os.path.exists(MICRO_CSV), 'Ensure microscopy csv exists'

    # Get in situ data
    get_predicted_insitu_images()



    # Create density csv file

if __name__ == '__main__':
    main()