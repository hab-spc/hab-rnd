import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy.stats import entropy, poisson
from sklearn.metrics import mean_absolute_error
from statsmodels.formula.api import glm


def fit_distribution(tf_col, data, distribution, pre):
    """Fit distribution"""

    def get_alpha_value(count, data):
        expr = '{cc} ~ {cc}'.format(cc=count)
        poisson_model = glm(expr, data=data, family=sm.families.Poisson()).fit()
        data['BB_LAMBDA'] = poisson_model.mu  # lambda value

        data['AUX_OLS_DEP'] = data.apply(lambda x: ((x[count] - x['BB_LAMBDA']) ** 2 - x[
            count]) / x['BB_LAMBDA'], axis=1)
        ols_expr = """AUX_OLS_DEP ~ BB_LAMBDA - 1"""
        aux_olsr_results = smf.ols(ols_expr, data).fit()

        return aux_olsr_results.params[0]

    predicted_cols = []
    logger = logging.getLogger(__name__)
    for col in tf_col:
        # Get data
        expr = '{cc} ~ {cc}'.format(cc=tf_col[col])
        y, x = dmatrices(expr, data=data, return_type='dataframe')

        # Instantiate model
        if 'zinflate poisson' in pre:
            model = sm.ZeroInflatedPoisson(y, x, inflation='logit')
            # model = reg_model.ZeroInflatedPoisson(y, x, x, inflation='logit')
        else:
            if 'nbinom' in pre:
                alpha = get_alpha_value(tf_col[col], data)
                distribution = sm.families.NegativeBinomial(alpha=alpha)

            model = glm(expr, data=data, family=distribution)

        # Fit the model
        if 'qpoisson' in pre:
            model = model.fit(cov_type='HC1')
        else:
            model = model.fit(method="nm", maxiter=50)

        # Predict the data
        try:
            predicted = model.get_prediction(x)
            predicted_counts = predicted.summary_frame()['mean']
        except:
            # predicted_counts = model.predict(x)
            zip_mean_pred = model.predict(x, exog_infl=np.ones((len(x), 1)))
            predicted_counts = poisson.ppf(q=0.95, mu=zip_mean_pred)

        # Test distributions
        actual_counts = data[tf_col[col]]
        kl_div = kl_divergence(actual_counts, predicted_counts)
        mae = mean_absolute_error(actual_counts, predicted_counts)

        logger.info(f'Fitted distribution results\n{"-" * 30}')
        logger.info(f'KL Div: {kl_div}')
        logger.info(f'MAE: {mae}')
        logger.info(model.summary())

        # Save data
        predicted_cols.append(pre + col.replace('_', ' '))
        data[pre + col.replace('_', ' ')] = predicted_counts

    return data[predicted_cols]


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
