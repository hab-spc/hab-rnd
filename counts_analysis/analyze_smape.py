import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_smape_vs_class_data(data, micro_counts, lab_counts, pier_counts):
    """Get smape vs class data for plotting"""

    def get_smape_statistics(y_true, y_pred):
        # give statistics for each class (median, 95CI, std dev) of each setting
        denominator = (np.abs(y_true) + np.abs(y_pred))
        diff = np.abs(y_true - y_pred) / denominator * 100.0
        if math.isnan(diff):
            return 0
        else:
            return diff

    data['micro_lab_smape'] = data.apply(
        lambda x: get_smape_statistics(x[micro_counts], x[lab_counts]), axis=1)
    data['micro_pier_smape'] = data.apply(
        lambda x: get_smape_statistics(x[micro_counts], x[pier_counts]), axis=1)
    data['lab_pier_smape'] = data.apply(
        lambda x: get_smape_statistics(x[lab_counts], x[pier_counts]), axis=1)
    data = data.sort_values(by='class')
    return data


def plot_smape_class_characterization(data, micro, lab, pier):
    """Plot smape class characterization"""
    smape_df = get_smape_vs_class_data(data, micro, lab, pier)
    fig, ax = plt.subplots(1, 3)

    sns.barplot(x='class', y='micro_lab_smape', data=smape_df, capsize=.1)
    plt.xticks(rotation=90);
    plt.ylabel('SMAPE');
    plt.title('Micro vs Lab Class SMAPE Scores')
    plt.show()

    sns.barplot(x='class', y='micro_pier_smape', data=smape_df, capsize=.1)
    plt.xticks(rotation=90);
    plt.ylabel('SMAPE');
    plt.title('Micro vs Pier Class SMAPE Scores')
    plt.show()

    smape_df = smape_df.sort_values(by='class')
    sns.barplot(x='class', y='lab_pier_smape', data=smape_df, capsize=.1)
    plt.xticks(rotation=90);
    plt.ylim(0, 100);
    plt.ylabel('SMAPE');
    plt.title('Lab vs Pier Class SMAPE Scores')
    plt.show()
