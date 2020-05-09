import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy.stats import entropy, poisson
from sklearn.metrics import mean_absolute_error
from statsmodels.formula.api import glm


def fit_distribution(count_col, data, distributions):
    logger = logging.getLogger(__name__)

    predicted_df = pd.DataFrame()
    fitted_scores = pd.DataFrame()
    for dist in distributions:
        actual_counts = data[count_col]
        # get predicted counts
        pred_counts, fit_score = get_predicted_counts(actual_counts, dist)
        # test predicted columns
        fit_score['distribution'] = dist
        fitted_scores = fitted_scores.append(fit_score, ignore_index=True)
        # save predicted counts under tsfmed sample method
        predicted_df[dist + " " + count_col] = pred_counts

    logger.info(f'\nDistributions sorted by goodness of fit:\n{"-" * 40}')
    logger.info(fitted_scores)
    logger.info('\n')

    return predicted_df


def get_predicted_counts(data, dist):
    """Fit distribution"""
    col_name = data.name.replace(' ', '_').replace('/', '_')
    data = data.rename(col_name)

    expr = '{cc} ~ {cc}'.format(cc=col_name)

    def get_alpha_value(count, data):
        poisson_model = glm(expr, data=data, family=sm.families.Poisson()).fit()
        data['BB_LAMBDA'] = poisson_model.mu  # lambda value

        data['AUX_OLS_DEP'] = data.apply(lambda x: ((x[count] - x['BB_LAMBDA']) ** 2 - x[
            count]) / x['BB_LAMBDA'], axis=1)
        ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
        aux_olsr_results = smf.ols(ols_expr, data).fit()

        return aux_olsr_results.params[0]

    # Get data
    data_ = data.to_frame()
    y, actual_counts = dmatrices(expr, data=data_, return_type='dataframe')

    # Instantiate model
    if 'zinflate poisson' in dist:
        model = sm.ZeroInflatedPoisson(y, actual_counts, inflation='logit')
        # model = reg_model.ZeroInflatedPoisson(y, x, x, inflation='logit')
    else:
        if 'nbinom' in dist:
            alpha = get_alpha_value(col_name, data_)
            distribution = sm.families.NegativeBinomial(alpha=alpha)
        elif 'poisson' in dist:
            distribution = sm.families.Poisson()

        model = glm(expr, data=data_, family=distribution)

    # Fit the model
    try:
        if 'qpoisson' in dist:
            model = model.fit(cov_type='HC1')
        else:
            model = model.fit(maxiter=50)
    except:
        model = model.fit(method='nm')

    # Predict the data
    try:
        predicted = model.get_prediction(actual_counts)
        predicted_counts = predicted.summary_frame()['mean']
    except:
        try:
            predicted_counts = model.predict(actual_counts)
        except:
            zip_mean_pred = model.predict(actual_counts,
                                          exog_infl=np.ones((len(actual_counts), 1)))
            predicted_counts = poisson.ppf(q=0.95, mu=zip_mean_pred)

    # test fit scores
    fit_scores = test_goodness_fit(data, predicted_counts, model)

    return predicted_counts, fit_scores


def test_goodness_fit(actual_counts, predicted_counts, model):
    try:
        kl_div = kl_divergence(actual_counts, predicted_counts)
        mae = mean_absolute_error(actual_counts, predicted_counts)
        pchi = model.pearson_chi2
        dev = model.deviance
        st_sig = 268.531 < pchi

        return {'kl div': kl_div, "mae": mae, 'pearson chi2': pchi, 'deviance': dev,
                'statistically significant': st_sig}
    except:
        return {'kl div': kl_div, "mae": mae}


def print_stats(data):
    def get_confidence_limit(stats):
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(100.0, np.percentile(stats, p))
        return alpha * 100, lower, upper

    print(f'COUNT: {len(data)}')
    print(f'AVG: {np.mean(data)}')
    print(f'MEDIAN: {np.median(data)}')
    print(f'STD DEV: {np.std(data)}')
    print(f'VAR: {np.var(data)}')
    print('%.1f confidence interval %.2f%% and %.2f%%\n' % (get_confidence_limit(data)))


def plot_distribution(data, verbose=True):
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True,
                                          gridspec_kw={"height_ratios": (0.2, 1)})

    mean = np.mean(data)
    median = np.median(data)

    # boxplot
    sns.boxplot(data, ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')

    sns.distplot(data, kde=False, ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')

    plt.legend({'Mean': mean, 'Median': median})
    if verbose:
        print_stats(data)
    plt.show()


def kl_divergence(p, q):
    eps = 0.01
    pp = p + eps
    pp /= sum(pp)

    qq = q + eps
    qq /= sum(qq)
    kl_div = entropy(pp, qq, base=10.)
    return kl_div
