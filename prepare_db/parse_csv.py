""" """
import json
# Standard dist imports
import logging
import os

import numpy as np
# Third party imports
import pandas as pd

# Project level imports
from prepare_db.logger import Logger


# Module level constants

def get_time_density(data, time_col, time_freq, num_of_classes=2,
                     labels_uploaded=False, plot=False, insitu=False, save_dir=None):
    """Filter sampling time within dataframe

    Use case: filter a days worth of data down to 3 hour windows around
    the microscopy time (1 hour before, 1 hour interim, 1 hour after)

        Function currently groups estimates by the day. We need to be
        bale to do this for multiple time options (i.e. 5 minute
        estimates, 10 minute estimates, etc.)

    #TODO get rid of all string hard-codings in the script

    Args:
        data (pd.Dataframe): Dataframe to filter
        time_col (str): Name of time column to filter

    Returns:

    """
    df, logger = SPCParser.initialize_parsing(data, task_name='get-time-density')

    def parse_timestamp(x):
        """Parses example fmt: 'Sat Dec 23 10:01:24 2017 PST' """
        return ' '.join(x.split(' ')[1:-1])

    # Convert time column as datetime object
    if insitu:
        df[time_col] = df[time_col].apply(parse_timestamp)
    df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)

    # Group data by dates
    df['Date'] = df[time_col].dt.date
    grouped_dates_df = df.groupby('Date')

    # Get time density for each date and store back into dataframe
    time_density_df = pd.DataFrame()
    for ii, date_df in grouped_dates_df:
        logger.debug('=' * 25 + f' Time Stamp: {ii} ' + '=' * 25)

        # Calculate time bins
        maxTime, minTime, bins = SPCParser.calculate_time_bins(
            data=date_df, time_col=time_col, time_freq=time_freq)
        logger.debug(f'Start time: {minTime} | End time: {maxTime}')

        # Group the predictions w.r.t to each time bin
        grouped_time_bins = date_df.groupby(pd.cut(date_df[time_col], bins))['pred']

        # Create new dataframe for holding the grouped predictions
        dff = SPCParser.transpose_prediction_counts(grouped_time_bins=grouped_time_bins,
                                                    num_of_classes=num_of_classes,
                                                    save_dir=save_dir, insitu=insitu)

        # Calculate average and std dev
        dd = {time_col: ii}
        pre = 'pier_' if insitu else 'lab_'
        total_bins = grouped_time_bins.ngroups
        for cls_idx in range(0, num_of_classes):
            pred = dff[cls_idx].describe()
            dd[pre + str(cls_idx) + '_avg_{}'.format(time_freq)] = pred['mean']
            dd[pre + str(cls_idx) + '_std_{}'.format(time_freq)] = pred['std']

        def print_full(x):
            pd.set_option('display.max_rows', len(x))
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 2000)
            pd.set_option('display.float_format', '{:20,.2f}'.format)
            pd.set_option('display.max_colwidth', -1)
            logger.debug(x)
            pd.reset_option('display.max_rows')
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')
            pd.reset_option('display.float_format')
            pd.reset_option('display.max_colwidth')

        logger.debug('Total time bins: {}\n'.format(total_bins))
        print_full(dff.describe())
        time_density_df = time_density_df.append(pd.Series(dd), ignore_index=True)

    time_density_df = time_density_df.fillna(0)

    logger.info('Generated Dataframe')
    logger.info(time_density_df.head())
    return time_density_df


def get_lab_data(date_dir=None):
    """ Get lab dataframe given a lab imaging directory date

    Contains image abs path and any previous predictions generated for the lab image directory.
    Also extracts image date and time for research purposes

    Args:
        date_dir (str): Abs path to the date directory
            Examples: `/data6/phytoplankton-db/hab_in_vitro/images/20190520`

    Returns:
        pd.DataFrame: merged dataframe between meta data and predictions

    """
    logger = logging.getLogger('get_lab_data')
    data_file = os.path.join(date_dir, '{}')
    try:
        # Read in files and preprocess
        label_col = 'label'
        pred = json.load(open(data_file.format('predictions.json'), 'rb'))
        meta_df = pd.read_csv(data_file.format('features.tsv'), sep='\t')
    except:
        try:
            meta_df = pd.read_csv(data_file.format('features.csv'))
        except:
            logger.error(f'ERROR: Meta data not found for {date_dir}')
            return None

    # Create image id to conduct merging
    meta_df = meta_df[meta_df.columns.values[:11]]
    meta_df['image_id'] = meta_df['url'].apply(
        lambda x: os.path.basename(x).replace('.jpeg', '.tif'))

    # Preprocess prediction json
    pred_df = pd.DataFrame(pred['machine_labels'])
    pred_df = pred_df.rename({'gtruth': label_col}, axis=1)

    # Merge based off image_id
    merged = meta_df.merge(pred_df, on='image_id')
    if pred_df.shape[0] != meta_df.shape[0]:
        logger.warning(
            'Inconsistency between meta and predictions {} vs {} for date {}'.format(meta_df.shape[0], pred_df.shape[0],
                                                                                     date_dir))

    # Extract image date and times
    # use case is more for sampling correlation plots
    merged['image_timestamp'] = pd.to_datetime(merged['timestamp'])
    merged['image_date'] = merged['image_timestamp'].dt.date
    merged['image_time'] = merged['image_timestamp'].dt.time
    merged = merged.drop(['timestamp'], axis=1)

    # Add deployment required columns and create abspath to images
    merged['user_labels'] = '[]'
    merged['images'] = merged['url'].apply(lambda x: os.path.join(date_dir, 'static', x))
    merged = merged.drop('url', axis=1)

    return merged


class SPCParser(object):
    """Parse SPC image streaming api data

    Disregard the script filename. Need to figure out
    correct place to put this and name it.

    ALSO this is more for validating correlation graphs atm

    """

    def __init__(self, csv_fname=None, json_fname=None, classes=None, save=False):
        """ Initializes SPCParsing instance to extract data from dataset

        Args:
            csv_fname (str): Abs path to the csv file (dataset)
            json_fname (str): Abs path to the json file containing predictions
            classes (str): Abs path to the training.log that contains the classes
            save (bool): Flag to save the merged dataset
        """
        if csv_fname:
            self.csv_fname = csv_fname
            self.dataset = pd.read_csv(csv_fname)

        if json_fname:
            self.json = json.load(open(json_fname, 'rb'))

        if csv_fname and json_fname:
            self.dataset = self.merge_dataset(save=save)

        if classes:
            self.cls2idx, self.idx2cls = self.get_classes(classes)

        # Get the hab species of interest for class filtering
        self.hab_species = open('/data6/phytoplankton-db/hab.txt', 'r').read().splitlines()

    def merge_dataset(self, save=False):
        # Drop outdated `label` column (used as gtruth in machine learning exp)
        label_col = 'label'
        meta_df = self.dataset.copy()
        meta_df = meta_df.rename({'timestamp': 'image_timestamp'})

        # Preprocess prediction json
        pred_df = pd.DataFrame(self.json['machine_labels'])
        pred_df = pred_df.rename({'gtruth': label_col}, axis=1)
        pred_df['image_id'] = pred_df['image_id'].apply(SPCParser.extract_img_id)

        # Merge based off image_id
        merged = meta_df.merge(pred_df, on='image_id')

        # Overwrite previous csv file with new gtruth
        if save:
            csv_fname = self.csv_fname.split('.')[0] + '-predictions.csv'
            print(f'Saved as {csv_fname}')
            merged.to_csv(csv_fname, index=False)
        return merged

    def get_gtruth(self, gtruth_col='label', verbose=False):
        """Get the gtruth distributions"""
        if verbose:
            print(self.dataset[gtruth_col].value_counts())
        self.gtruth = self.dataset[gtruth_col].tolist()

    def get_predictions(self, pred_col='pred', verbose=False):
        """Get the prediction distribution"""
        if verbose:
            print(self.dataset[pred_col].value_counts())
        self.pred = self.dataset[pred_col].tolist()

    def get_classes(self, filename):
        """Set class2idx, idx2class encoding/decoding dictionaries"""
        class_list = SPCParser._parse_classes(filename)
        cls2idx = {i: idx for idx, i in enumerate(sorted(class_list))}
        idx2cls = {idx: i for idx, i in enumerate(sorted(class_list))}
        return cls2idx, idx2cls

    def filter_classes(self, micro_data):
        """Take out any NaN, nonHAB and untrained classes"""
        # drop nan classes
        df1 = micro_data.copy()
        nan_classes = df1.columns[df1.isna().all()].tolist()
        df1 = df1.drop(nan_classes, axis=1)
        print(f'NaN classes dropped: {sorted(nan_classes)}')

        # filter nonHAB classes
        hab = sorted(list(set(df1.columns).intersection(self.hab_species)))
        df1 = df1[list(micro_data.columns[:3]) + hab]
        print(f'NonHAB classes dropped: {set(self.hab_species).difference(hab)}')

        # combine any classes
        trained_classes = sorted(list(set(self.cls2idx.keys()).intersection(set(df1.columns))))
        print(f'Untrained classes dropped: {set(df1.columns.difference(trained_classes))}')
        df1 = df1[list(micro_data.columns[:3]) + trained_classes]

        trained_classes = {i: self.cls2idx[i] for i in trained_classes}
        pre = 'micro_{}'
        for k, v in trained_classes.items():
            df1.rename({k: pre.format(v)}, axis=1, inplace=True)

        return df1, trained_classes

    @staticmethod
    def _parse_classes(filename):
        """Parse MODE_data.info file"""
        lbs_all_classes = []
        with open(filename, 'r') as f:
            label_counts = f.readlines()
        label_counts = label_counts[:-1]
        for i in label_counts:
            class_counts = i.strip()
            class_counts = class_counts.split()
            class_name = ''
            for j in class_counts:
                if not j.isdigit():
                    class_name += (' ' + j)
            class_name = class_name.strip()
            lbs_all_classes.append(class_name)
        return lbs_all_classes

    @staticmethod
    def extract_top_k(x):
        prob = eval(x)

    @staticmethod
    def extract_img_id(x):
        return os.path.basename(x).split('.')[0] + '.tif'

    @staticmethod
    def initialize_parsing(data, task_name=None):
        """Creates copy of dataframe and returns logger
                LOGGING ITEMS during csv creation
        """
        df = data.copy()
        logger = logging.getLogger(task_name)
        logger.setLevel(logging.INFO)
        Logger.section_break(task_name)
        return df, logger

    @staticmethod
    def distribution_report(df, column, logger):
        """Writes analysis of data distribution to log
                LOGGING ITEMS during csv creation

        """
        logger.debug('{} distribution | size: {}'.format(column, df.shape))
        logger.debug(df[column].value_counts())
        logger.debug('\n')

    @staticmethod
    def get_gtruth_counts(data):
        """Compute gtruth and corrected classifier abundance estimates

        Applicable only if gtruth is available

        """
        df, logger = SPCParser.initialize_parsing(data, 'get-cell-count')
        pre = 'clsfier_'
        df[pre + 'Prorocentrum'] = df['Prorocentrum'] + df['False Prorocentrum']
        df[pre + 'Non-Prorocentrum'] = df['Non-Prorocentrum'] + df['False Non-Prorocentrum']

        df = df.drop(['False Prorocentrum', 'False Non-Prorocentrum'], axis=1)
        pre = 'corrected_'
        df = df.rename(columns={'Prorocentrum': pre + 'Prorocentrum',
                                'Non-Prorocentrum': pre + 'Non-Prorocentrum'})
        logger.debug('Added {}'.format(set(df.columns).difference(data.columns)))
        logger.debug(df.head())
        return df

    @staticmethod
    def compute_imaged_volume(class_size=0.07):
        """Compute image volume"""
        min_samp = 20
        min_pixels_per_obj = min_samp

        class_size = 1000*np.array([class_size])
        # size_classes = np.linspace(1.0, 1000.0, 100000)

        min_resolution = class_size / min_pixels_per_obj
        pixel_size = min_resolution / 2
        blur_factor = 3
        wavelength = 0.532
        NA = 0.61 * wavelength / min_resolution
        NA[NA >= 0.75] = 0.75

        div_angle = np.arcsin(NA)
        img_DOF = blur_factor * pixel_size / np.tan(div_angle) / 2

        # compute the imaged volume in ml
        imaged_vol = pixel_size ** 2 * 4000 * 3000 * img_DOF / 10000 ** 3
        return imaged_vol

    @staticmethod
    def calculate_time_bins(data, time_col, time_freq):
        # Get minimum and maximum prior to binning
        minTime = data[time_col].min()
        maxTime = data[time_col].max()
        deltaT = pd.Timedelta(time_freq)

        # Calculate time range w.r.t to bin
        minTime -= deltaT - (maxTime - minTime) % deltaT
        bins = pd.date_range(start=minTime, end=maxTime, freq=time_freq)
        return maxTime, minTime, bins

    @staticmethod
    def transpose_prediction_counts(grouped_time_bins, num_of_classes, save_dir, insitu=False):
        # Create new dataframe for holding the grouped predictions
        dff = pd.DataFrame(columns=['time'] + list(range(0, num_of_classes)))

        # Traverse through groups and store counts into new dataframe
        # dates by predicted class count matrix
        for idx, (i, tt) in enumerate(grouped_time_bins):
            d = np.zeros((1, num_of_classes))
            tt_dict = tt.value_counts().to_dict()
            for k, v in tt_dict.items():
                d[:, k] = v
            ll = [i] + d[0, :].tolist()
            dff.loc[idx] = ll

        if save_dir:
            date_fname = i.left._date_repr + '{}_pred_counts.csv'.format('in_situ' if insitu else 'in_vitro')
            abs_dir = os.path.join(save_dir, 'counts')
            if not os.path.exists(abs_dir):
                os.makedirs(abs_dir)
            dff.to_csv(os.path.join(abs_dir, date_fname), index=False)

        return dff
