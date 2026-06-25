Corporate Credit-Rating Prediction — XGBoost vs TabPFN-3 vs AutoGluon
Predicting corporate credit ratings (10 broad classes, D … AAA) from 16 financial-ratio features, comparing three tabular approaches:

Model
Paradigm
XGBoost
manually-tuned gradient boosting (baseline)
TabPFN-3
zero-shot pre-trained tabular transformer
AutoGluon
AutoML stack-ensemble


The repository accompanies the thesis. Core logic lives in the credit_rating Python package; the notebooks are thin, documented wrappers that call it.
Repository structure
credit-rating-prediction/

├── README.md

├── requirements.txt

├── pyproject.toml            # enables `pip install -e .`

├── credit_rating/            # all reusable code (the *.py files)

│   ├── config.py             # paths, constants, device, TabPFN token

│   ├── data.py               # load, preprocess, company-aware split

│   ├── models.py             # model factories + class padding

│   ├── evaluate.py           # metrics, per-class report, confusion/importance plots

│   ├── experiments.py        # XGBoost / TabPFN sweep / AutoGluon grid / stability / learning curve

│   └── plots.py              # sweep, grid heatmap, stability, learning-curve figures

├── notebooks/                # run top-to-bottom, in order

│   ├── 1_data_exploration.ipynb   # load, preprocess, build & save the split

│   ├── 2_training.ipynb           # train 3 models + hyperparameter analyses + plots

│   └── 3_results.ipynb            # comparison, stability test, learning curve

├── data/                     # put the raw CSV here (not committed)

└── results/                  # splits, *.pkl, CSVs, and figures/ (generated)
Installation
Python 3.10+ recommended (GPU strongly recommended for TabPFN-3).

git clone <this-repo-url>

cd credit-rating-prediction

python -m venv .venv && source .venv/bin/activate   # optional

pip install -r requirements.txt

pip install -e .            # makes `import credit_rating` work everywhere

Then place the dataset at data/corporateCreditRatingWithFinancialRatios.csv (see data/README.md).
TabPFN-3 license token
TabPFN-3 needs a one-time Prior Labs token to download its weights for local inference. Get one at https://ux.priorlabs.ai (register → accept the license → copy the API key), then make it available in one of these ways:

export TABPFN_TOKEN="<your-api-key>"     # local / shell

On Google Colab, store it as a Secret named TABPFN_TOKEN (key icon in the left sidebar) and switch on Notebook access. models.make_tabpfn picks it up automatically.
Running on Google Colab
Mount Drive and point the package at your data folder before importing:

from google.colab import drive; drive.mount('/content/drive')

import os

os.environ["THESIS_DATA_DIR"]    = "/content/drive/MyDrive/THESIS/clean_redo"

os.environ["THESIS_RESULTS_DIR"] = "/content/drive/MyDrive/THESIS/clean_redo"

!pip install -q tabpfn autogluon xgboost
Usage
Run the notebooks in order:

1_data_exploration.ipynb — loads and preprocesses the data, inspects the class distribution, and writes the company-aware split to results/splits.pkl.
2_training.ipynb — trains all three models and runs each hyperparameter study: the XGBoost baseline, the TabPFN-3 n_estimators sweep, and the AutoGluon 2×2 grid search (num_bag_folds × num_stack_levels). Produces the per-class reports, confusion matrices, sweep/marginal-gain plots, grid heatmaps, and feature-importance charts, and saves each model's results.
3_results.ipynb — builds the cross-model comparison, then runs the stability analysis (20 group-splits + Nadeau–Bengio corrected t-test) and the learning curve (F1 vs training size over 5 seeds).

Or run the whole pipeline end-to-end without opening the notebooks:

python run_all.py                              # full run (slow)

python run_all.py --skip-autogluon --quick     # fast smoke test

python run_all.py --stages data xgboost tabpfn # selected stages only

Everything is also callable directly:

from credit_rating import data, experiments

X_train, X_test, y_train, y_test = data.prepare_and_save_splits()

sweep = experiments.tabpfn_n_estimators_sweep(X_train, X_test, y_train, y_test)

Runtime note. The AutoGluon grid search, stability analysis, and learning curve are slow. The stability and learning-curve cells cache to CSV and reload on re-run; pass run_ag=False to those experiment functions for a quick pass without AutoGluon.
Method notes
Group-aware split on the company column prevents the same firm appearing in both train and test (avoids leakage from memorised firm profiles).
Preprocessing: notched ratings collapsed to broad classes, features winsorised at the 1st/99th percentile, ratings encoded D=0 … AAA=9.
Selection: the AutoGluon grid is selected on internal  test_f1_macro

