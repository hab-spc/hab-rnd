# hab_rnd
Research side of the hab projects

## Set up

### System Requirements
1. Python 3.6 or higher
2. Python libraries: pandas, holoviews, scipy, seaborn, hvplot, jupyterlab

#### Conda Setup Instructions
If you are using conda, you can get started by cloning this repository and using the 
environment.yaml file as in the following:
```bash
conda env create -n hab_rnd -f env.yml
```
and then activate the environment using the following,
```bash
conda activate hab_rnd
```

Next add the environment to the jupyter kernels:
```bash
python -m ipykernel install --user --name=hab_rnd
```
Once you run `jupyter lab` you should find the hab_rnd virtual environment as one of your notebook kernel options in
 the launcher.

### Download required files
Download the [dataset](https://drive.google.com/file/d/183opFeIupbHUN-6Ifq7jBcfwV6JNSSk8/view?usp=sharing) and store
 it within the project dir as `phytoplankton-db`.
The main dataset for comparing counts is under `../counts`. Use the latest version.

Next, ensure that the root directory variable in `counts_analysis/c_utils.py` for accessing the datasets is set as the
 following:
```
ROOT_DIR = 'phytoplankton-db'

COUNTS_CSV = {
    'counts': f'{ROOT_DIR}/counts/master_counts_v11.csv',
    # truncate pier to 1000s (500s offset)
    'counts-pier1000s': f'{ROOT_DIR}/counts/master_counts-pier1000s.csv',
    # include other counts for the time series
    'counts-v9': f'{ROOT_DIR}/counts/master_counts_v9.csv',  # Baseline
    'counts-v10': f'{ROOT_DIR}/counts/master_counts_v10.csv',  # CV model
    # zhouyuan baseline model
    'tsfm-counts': f'{ROOT_DIR}/counts/master_counts_v4-tsfm.csv'
}
```
### Run a sample analysis


## Directory Structure
The directory structure of the Hab_RND looks like this:
```.env
├── README.md
├── analyze_poor_performance.ipynb          <- One-time notebook for performance analysis
├── auto_pier_analysis_notebooks
│   ├── Auto-Pier_Analysis.ipynb            <- 6 Hour Fill-in-the-Blanks Auto-Pier analysis
│   └── Auto-Pier_Analysis_34min.ipynb      <- 34 min Fill-in-the-Blanks Auto-Pier analysis
├── cells_ml_time_series.ipynb              <- One-time notebook for microscopy analysis
├── class_counts-summary.ipynb              
├── classifier_work.ipynb
├── counts_analysis
│   ├── analyze_smape.py
│   ├── c_utils.py
│   ├── calibrate_metric-zhouyuan.py
│   ├── calibrate_metric.py
│   ├── class_counts.py
│   ├── gtruth_analysis.py
│   ├── investigate_metric.py
│   ├── investigate_temporal_vol_error.py
│   ├── plot_class_error_counts.py
│   ├── plot_class_summary.py
│   ├── plot_daily_counts_summary.py
│   ├── plot_reldist_time.py
│   └── time_smape_analysis.py
├── cv_pier_confusion_matrix.png
├── daily_counts-summary.ipynb
├── eval_classifier_count_error.py
├── eval_counts.py                          <- Script to generate count error comparisons
├── eval_error_metric.py                    <- Script to compare agreement & error metrics
├── experiment_metric.py
├── figures.ipynb
├── get_counts.py                           <- Script to generate counts datasets
├── get_roi_counts.py
├── mase_analysis_raw_counts.ipynb
├── methods-summary.ipynb                   <- Main notebook for comparing methods & some figure generations
├── prepare_db
│   ├── bootstrap_counts.py
│   ├── create_csv.py
│   ├── logger.py
│   ├── parse_csv.py
│   └── set_micro_counts.py
├── recover_counts.py
├── relative_abundance_work-all_classes.ipynb
├── requirements.txt
├── results__counts_analysis.ipynb
└── validate_exp                            <- Old directory for validation comparisons
    ├── __init__.py
    ├── get_predicted_day_counts.py
    ├── get_seq_timebins_counts.py
    ├── get_timewindow_counts.py
    ├── stat_fns.py                         <- Script of statistical analysis functions
    ├── transform_data.py
    ├── v_utils.py
    ├── val_go.py
    └── validate_time_dst.ipynb
```