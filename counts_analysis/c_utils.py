import numpy as np
import seaborn as sns
from scipy.stats import entropy

COUNTS_CSV = {
    'counts': '/data6/phytoplankton-db/counts/master_counts_v7.csv',
    # truncate pier to 1000s (500s offset)
    'counts-pier1000s': '/data6/phytoplankton-db/counts/master_counts-pier1000s.csv',
    # include other counts for the time series
    'counts-v9': '/data6/phytoplankton-db/counts/master_counts_v9.csv',  # Baseline
    'counts-v10': '/data6/phytoplankton-db/counts/master_counts_v10.csv',  # CV model
    # zhouyuan baseline model
    'tsfm-counts': '/data6/phytoplankton-db/counts/master_counts_v4-tsfm.csv'
}
MODEL_DIR = '/data6/yuanzhouyuan/hab/hab-ml/experiments/baseline_new_weighted_loss'
CV_MODEL_DIR = '/data6/phytoplankton-db/models'
IMG_CSV = {
    'lab': '/data6/phytoplankton-db/csv/hab_in_vitro_summer2019.csv',
    'pier': '/data6/phytoplankton-db/csv/hab_in_situ_summer2019.csv',
    'lab-pred': f'{MODEL_DIR}/hab_in_vitro_summer2019-predictions.csv',
    'pier-pred': f'{MODEL_DIR}/hab_in_situ_summer2019-predictions.csv',
    'lab-cv-pred': f'{CV_MODEL_DIR}/cv_hab_in_vitro_summer2019-predictions.csv',
    'pier-cv-pred': f'{CV_MODEL_DIR}/cv_hab_in_situ_summer2019-predictions.csv',
}

# Classes
CLASSES = ['Akashiwo',
           'Ceratium falcatiforme or fusus',
           'Ceratium furca',
           'Chattonella',
           'Cochlodinium',
           'Lingulodinium polyedra',
           'Prorocentrum micans',
           'Pseudo-nitzschia chain'
           ]

CORRELATED_CLASSES = [
    'Lingulodinium polyedra',
    'Prorocentrum micans'
]


def set_counts(label, counts, micro_default=True):
    micro_counts = 'micro {}'.format('cells/mL' if micro_default else counts)
    lab_counts = f'lab {label} {counts}'
    pier_counts = f'pier {label} {counts}'
    return micro_counts, lab_counts, pier_counts


def set_settings(counts):
    score_settings = {'lab - micro': (counts[0], counts[1]),
                      'pier - micro': (counts[0], counts[2]),
                      'pier - lab': (counts[1], counts[2])}
    return score_settings


def counts_meta(counts): return ['class', 'datetime'] + list(counts)

def get_units(smpl_technique):
    if 'pier' in smpl_technique:
        if 'nrmlzd raw count' in smpl_technique:
            return 'cells/1000s'
        elif 'cells/mL' in smpl_technique:
            return 'cells/mL'
        else:
            return 'cells/2000s'

    elif 'lab' in smpl_technique:
        if 'cells/mL' in smpl_technique:
            return 'cells/mL'
        else:
            return 'cells/1000s'

    else:
        return 'cells/mL'


def set_color_map():
    current_palette_7 = sns.color_palette("coolwarm", 9)
    sns.set_palette(current_palette_7)


def smape(y_true, y_pred):
    return 100.0 / len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true)
                                                                   + np.abs(y_pred)))


def kl_divergence(p, q):
    eps = 0.01
    pp = p + eps
    pp /= sum(pp)

    qq = q + eps
    qq /= sum(qq)
    kl_div = entropy(pp, qq, base=10.)
    return kl_div
