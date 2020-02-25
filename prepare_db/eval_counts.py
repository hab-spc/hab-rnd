

def create_eval_csv(input_dir):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from validate_exp.v_utils import concordance_correlation_coefficient, smape
    from scipy.spatial import distance

    # Initialize logging
    log_fname = os.path.join(input_dir, 'eval_csv.log')
    Logger(log_fname, logging.INFO, log2file=False)
    Logger.section_break('Create EVAL CSV')
    logger = logging.getLogger('create-csv')

    data = pd.read_csv(os.path.join(input_dir, 'master_count.csv'))
    df = data.copy()

    """loop over for each sample method (lab & pier) and concatenate it to the main_df"""
    sample_method = 'pier'
    label = 'gtruth'
    temp_gtruth = df[df['label'] == 'gtruth']
    temp_gtruth = temp_gtruth.rename({f'{sample_method} total abundance': f'{sample_method} {label} total abundance',
                        'raw count': f'{sample_method} {label} raw count',
                        f'{sample_method} relative abundance': f'{sample_method} {label} relative abundance'}, axis=1)
    temp_gtruth = temp_gtruth.drop('label', axis=1)

    label = 'predicted'
    temp_pred = df[df['label'] == 'prediction']
    temp_pred = temp_pred.rename({f'{sample_method} total abundance': f'{sample_method} {label} total abundance',
                        'raw count': f'{sample_method} {label} raw count',
                        f'{sample_method} relative abundance': f'{sample_method} {label} relative abundance'}, axis=1)
    temp_pred = temp_pred.drop('label', axis=1)

    concat = temp_pred.merge(temp_gtruth, on=['class', 'datetime', 'micro raw count', 'micro relative abundance', 'micro total abundance'])

    # evaluation
    # loop over each class, take the mean absolute error and ccc, then store them into a data structure
    sample_method_gold_std = 'micro'
    sample_method_to_test = 'pier'
    label_type = 'gtruth'
    sample_method_gold_std_col = f'{sample_method_gold_std} relative abundance'
    sample_method_to_test_col = f'{sample_method_to_test} {label_type} relative abundance'

    eval_metrics = {'class': [], 'mae': [], 'mse': [], 'smape':[], 'ccc':[], 'bray curtis':[]} # 'class'; 'mean absolute error'; 'ccc'
    classes = sorted(concat['class'].unique())
    for cls in classes:
        # get data
        temp = concat.loc[concat['class'] == cls]
        smpl_gold_std, smpl_to_test = temp[sample_method_gold_std_col], temp[sample_method_to_test_col]

        eval_metrics['class'].append(cls)
        # MAE
        eval_metrics['mae'].append(mean_absolute_error(smpl_gold_std, smpl_to_test))
        # MSE
        eval_metrics['mse'].append(mean_squared_error(smpl_gold_std, smpl_to_test))
        # smape
        eval_metrics['smape'].append(smape(smpl_gold_std, smpl_to_test))
        # CCC
        eval_metrics['ccc'].append(concordance_correlation_coefficient(smpl_gold_std, smpl_to_test))
        # bray curtis
        eval_metrics['bray curtis'].append(distance.braycurtis(smpl_gold_std, smpl_to_test))

    eval_metrics['class'].append('Overall')
    eval_metrics['mae'].append(mean_absolute_error(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['mse'].append(mean_squared_error(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['smape'].append(smape(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['ccc'].append(concordance_correlation_coefficient(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))
    eval_metrics['bray curtis'].append(distance.braycurtis(concat[sample_method_gold_std_col], concat[sample_method_to_test_col]))

    # Create dataframe and save to csv
    eval_df = pd.DataFrame(eval_metrics)
    csv_fname = os.path.join(input_dir, 'eval.csv')
    eval_df.to_csv(csv_fname, index=False)
    logger.info(f'Saved eval csv as {csv_fname}')