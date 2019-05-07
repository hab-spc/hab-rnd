""" """
# Standard dist imports
import logging
import os
import re

# Third party imports
import pandas as pd
import numpy as np

# Project level imports
from utils.logger import Logger

# Module level constants

class SPCParser(object):
    """Parse SPC image streaming api data

    Disregard the script filename. Need to figure out
    correct place to put this and name it.

    ALSO this is more for validating correlation graphs atm

    """
    @staticmethod
    def create_img_id(x):
        return os.path.basename(x).split('.')[0] + '.tif'

    @staticmethod
    def clean_labels(data, label_col):
        """Specialized cleaning for Prorocentrum labels ONLY

        #TODO function to write up if needed when cleaning up labels from SPICI

        """
        #TODO clean up labels other than prorocentrums
        df, logger = SPCParser.initialize_parsing(data, 'clean-labels')

        # change all [] as Non-Prorocentrum
        # anything with a false prorocentrum, replace as a false-prorocentrum
        # get rid of everything else

        def clean_up(lbl, tag):
            if not lbl:
                if not tag:
                    return 'Unidentified'
                else:
                    return 'Non-Prorocentrum'
            elif 'False Prorocentrum' in lbl or \
                    'Prorocentrum_false_positiveal' in lbl:
                return 'False Prorocentrum'
            elif lbl[0] in ['Prorocentrum', 'False Non-Prorocentrum']:
                return lbl[0]
            else:
                return 'Non-Prorocentrum'

        df[label_col] = df.apply(lambda x: clean_up(x[label_col],
                                                    x['tags']), axis=1)
        logger.debug('After cleaning')
        SPCParser.distribution_report(df, label_col, logger)
        return df

    @staticmethod
    def extract_dateinfo(data, date_col, drop=True, time=False,
                         start_ref=pd.datetime(1900, 1, 1),
                         extra_attr=False):
        df = data.copy()

        # Extract the field
        def parse_timestamp(x):
            """Parses example fmt: 'Sat Dec 23 10:01:24 2017 PST' """
            return ' '.join(x.split(' ')[1:-1])

        fld = df[date_col].apply(parse_timestamp)

        # Check the time
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        # Convert to datetime if not already
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[date_col] = fld = pd.to_datetime(fld,
                                                infer_datetime_format=True)

        # Prefix for new columns
        pre = re.sub('[Dd]ate', '', '')
        pre = re.sub('[Tt]ime', '', pre)

        # Basic attributes
        attr = ['Date', 'Time', 'Year', 'Month', 'Week', 'Day', 'Dayofweek',
                'Dayofyear', 'Days_in_month', 'is_leap_year']

        # Additional attributes
        if extra_attr:
            attr = attr + ['Is_month_end', 'Is_month_start',
                           'Is_quarter_end',
                           'Is_quarter_start', 'Is_year_end',
                           'Is_year_start']

        # If time is specified, extract time information
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']

        # Iterate through each attribute
        for n in attr:
            df[n] = getattr(fld.dt, n.lower())

        # Calculate days in year
        df['Days_in_year'] = df['is_leap_year'] + 365

        if time:
            # Add fractional time of day (0 - 1) units of day
            df[pre + 'frac_day'] = ((df[pre + 'Hour']) + (
                    df[pre + 'Minute'] / 60) + (df[pre + 'Second'] / 60 / 60)) / 24

            # Add fractional time of week (0 - 1) units of week
            df[pre + 'frac_week'] = (df[pre + 'Dayofweek'] + df[
                pre + 'frac_day']) / 7

            # Add fractional time of month (0 - 1) units of month
            df[pre + 'frac_month'] = (df[pre + 'Day'] + (
                df[pre + 'frac_day'])) / (df[pre + 'Days_in_month'] + 1)

            # Add fractional time of year (0 - 1) units of year
            df[pre + 'frac_year'] = (df[pre + 'Dayofyear'] + df[
                pre + 'frac_day']) / (df[pre + 'Days_in_year'] + 1)

            # Add seconds since start of reference
        df[pre + 'Elapsed'] = (fld - start_ref).dt.total_seconds()

        if drop:
            df = df.drop(date_col, axis=1)

        return df

    @staticmethod
    def initialize_parsing(data, task_name=None):
        """Creates copy of dataframe and returns logger
                LOGGING ITEMS during csv creation
        """
        df = data.copy()
        logger = logging.getLogger(task_name)
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
    def get_time_density(data, time_col, time_bin, labels_uploaded=False):
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
        df, logger = SPCParser.initialize_parsing(data, task_name='filter-time')
        SPCParser.distribution_report(df, time_col, logger)

        label_col = 'user_labels'

        def parse_timestamp(x):
            """Parses example fmt: 'Sat Dec 23 10:01:24 2017 PST' """
            return ' '.join(x.split(' ')[1:-1])

        df[time_col] = df[time_col].apply(parse_timestamp)
        df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)

        df['Date'] = df[time_col].dt.date
        grouped_df = df.groupby('Date')
        new_df = pd.DataFrame()
        pre = 'clsfier_proro_'
        for ii, gr in grouped_df:

            #TODO change to make sure 5 min bins
            gr = gr.groupby(gr[time_col].dt.floor('{}'.format(time_bin)))
            dd = {time_col: ii}
            binned_times = []
            for i, grt in gr:

                # Get cell counts from either label or prediction col
                if labels_uploaded:
                    dd = gr[label_col].value_counts().to_dict()
                else:
                    binned_times.append(grt['predictions'].sum())

            # Grab other related info
            dd[pre+'avg_{}'.format(time_bin)] = np.mean(binned_times)
            dd[pre+'std_{}'.format(time_bin)] = np.std(binned_times)

            # Append dict to dataframe
            new_df = new_df.append(pd.Series(dd), ignore_index=True)

        new_df = new_df.fillna(0)

        logger.debug('After filtering')
        logger.debug(new_df.head())
        return new_df

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
    def compute_vol_density(x):
        img_vol = SPCParser.compute_imaged_volume(class_size=0.07)
        frame_rate = 8
        return x / (frame_rate * img_vol * 24*3600)

    @staticmethod
    def compute_time_density(x):
        """Convert image count into rate of cell/time"""
        return x

    @staticmethod
    def filter_annotator():
        pass

    """BEGIN ARCHIVED SCRAP"""
    @staticmethod
    def list_items2rows(df, column):
        """

        Drops unknown labels
        #TODO need to replace NaN values with unknown label

        Args:
            df: (pd.Dataframe)
            column: (str)

        Returns:
            pd.Dataframe

        """
        logger = logging.getLogger('list_item2rows')
        Logger.section_break('list_item2rows')
        df_ = df.copy()
        SPCParser.distribution_report(df_, column, logger)

        leftover_col = df_.columns.tolist()
        leftover_col.remove(column)
        df_[column] = df_[column].apply(lambda x: eval(x))
        df_ = df_[column].apply(pd.Series) \
            .merge(df_, right_index=True, left_index=True) \
            .drop([column], axis=1) \
            .melt(id_vars=leftover_col,
                  value_name=column[:-1]) \
            .drop("variable", axis=1) \
            .dropna()
        SPCParser.distribution_report(df_, column[:-1], logger)

        return df_

    @staticmethod #ARCHIVED SCRAP
    def filter_label(df, column, label):
        """Filters labels on a single basis

        assumes that each row is a single item rather than list like
        original data
        #ToDo how to handle false positive, false negative
        #ToDO how to handle multiple labels

        """
        df_ = df.copy()
        logger = logging.getLogger('filter_label')
        Logger.section_break('filter_label')
        logger.debug('Filtering by label {}'.format(label))
        SPCParser.distribution_report(df_, column, logger)
        df_ = df_[df_[column] == label].reset_index(drop=True)
        logger.debug('After filtering')
        SPCParser.distribution_report(df_, column, logger)

        return df_

    @staticmethod #ARCHIVED SCRAP
    def filter_labels(data, label_col, labels=['Prorocentrum']):
        df, logger = SPCParser.initialize_parsing(data,
                                                  task_name='filter-labels')
        SPCParser.distribution_report(df, label_col, logger)

        grouped_df = df.groupby(label_col)
        filtered_df = pd.DataFrame()
        for gr in grouped_df.groups.keys():
            if eval(gr) in labels:
                filtered_df = filtered_df.append(grouped_df.get_group(gr))
        filtered_df = filtered_df.reset_index(drop=True)
        logger.debug('After filtering')
        SPCParser.distribution_report(filtered_df, label_col, logger)
        return filtered_df
    """END: ARCHIVED SCRAP"""



