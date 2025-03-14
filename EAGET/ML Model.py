import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score
)
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import xgboost as xgb
import xlsxwriter
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    BaggingRegressor
)
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor

import smogn
import ImbalancedLearningRegression as iblr
from sklearn.model_selection import KFold

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('model_evaluation.log', mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def log_print(message, color=None, level=logging.INFO):
    try:
        logger.log(level, message)
    except UnicodeEncodeError:
        logger.log(level, message.encode('utf-8', errors='replace').decode('utf-8'))

RED = "\033[91m"
ENDC = "\033[0m"

def balance(train, strategy, params):
    log_print(f"Applying strategy: {strategy} with params: {params}")
    if train.isnull().values.any():
        log_print(f"{RED}Original training data has missing values, resampling aborted.{ENDC}", level=logging.ERROR)
        return None

    rare_condition = train['target'] > 10
    majority = train[~rare_condition]
    minority = train[rare_condition]

    if len(minority) == 0:
        log_print(f"{RED}No rare data with target > 10, resampling aborted.{ENDC}", level=logging.ERROR)
        return None

    try:
        if strategy == "SMT":
            train_resampled = iblr.smote(
                data=train,
                y='target',
                samp_method=params.get('C.perc', 'balance'),
                k=params.get('k', 5),
                rel_thres=0.8
            )
        elif strategy == "GN":
            train_resampled = iblr.gn(
                data=train,
                y='target',
                samp_method=params.get('C.perc', 'balance'),
                pert=params.get('pert', 0.1),
                rel_thres=0.8
            )
        elif strategy == "RO":
            train_resampled = iblr.ro(
                data=train,
                y='target',
                samp_method=params.get('C.perc', 'balance'),
                rel_thres=0.8
            )
        elif strategy == "RU":
            train_resampled = iblr.random_under(
                data=train,
                y='target',
                samp_method=params.get('C.perc', 'balance'),
                rel_thres=0.8
            )
        elif strategy == "SG":
            train_resampled = smogn.smoter(
                data=train,
                y='target',
                samp_method=params.get('samp_method', 'balance'),
                k=params.get('k', 5),
                pert=params.get('pert', 0.1),
                rel_xtrm_type='high',
                rel_thres=0.8
            )
        elif strategy == "WC":
            over = params.get('over', 0.5)
            under = params.get('under', 0.5)
            minority_over = minority.sample(frac=over, replace=True, random_state=42) if over > 0 else minority.copy()
            majority_under = majority.sample(frac=1 - under, random_state=42) if under > 0 else majority.copy()
            train_resampled = pd.concat([majority_under, minority_over], axis=0).reset_index(drop=True)
        else:
            log_print(f"{RED}Unknown resampling strategy: {strategy}{ENDC}", level=logging.ERROR)
            return None

        if train_resampled.isnull().values.any():
            log_print(f"{RED}Resampled data by {strategy} has missing values, trying to drop them.{ENDC}", level=logging.WARNING)
            train_resampled = train_resampled.dropna()
            if train_resampled.empty:
                log_print(f"{RED}Resampled data is empty, strategy: {strategy}, params: {params}{ENDC}", level=logging.ERROR)
                return None

        y_resampled_values = train_resampled['target'].unique()
        if len(y_resampled_values) < 2:
            log_print(f"{RED}Target variable has only one unique value after resampling, strategy: {strategy}{ENDC}", level=logging.ERROR)
            return None

        return train_resampled
    except Exception as e:
        log_print(f"{RED}Exception in balance function, error: {e}{ENDC}", level=logging.ERROR)
        return None

def evaluate_model_performance(model, X_train, y_train, X_test, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    evs_train = explained_variance_score(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r_train, _ = pearsonr(y_train, y_train_pred)
    huber_loss_train = np.mean(np.where(
        np.abs(y_train - y_train_pred) <= 1.0,
        0.5 * (y_train - y_train_pred) ** 2,
        1.0 * (np.abs(y_train - y_train_pred) - 0.5 * 1.0)
    ))

    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    evs_test = explained_variance_score(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r_test, _ = pearsonr(y_test, y_test_pred)
    huber_loss_test = np.mean(np.where(
        np.abs(y_test - y_test_pred) <= 1.0,
        0.5 * (y_test - y_test_pred) ** 2,
        1.0 * (np.abs(y_test - y_test_pred) - 0.5 * 1.0)
    ))

    metrics = {
        'model_name': model_name,
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'r2_train': r2_train,
        'r_train': r_train,
        'huber_loss_train': huber_loss_train,
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'r_test': r_test,
        'huber_loss_test': huber_loss_test
    }

    print(f"\nModel {model_name} training performance:")
    print(f"RMSE: {rmse_train:.4f}")
    print(f"MAE: {mae_train:.4f}")
    print(f"R²: {r2_train:.4f}")
    print(f"Pearson's R: {r_train:.4f}")
    print(f"Huber Loss: {huber_loss_train:.4f}")

    print(f"\nModel {model_name} testing performance:")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"MAE: {mae_test:.4f}")
    print(f"R²: {r2_test:.4f}")
    print(f"Pearson's R: {r_test:.4f}")
    print(f"Huber Loss: {huber_loss_test:.4f}")

    return metrics

def process_fold(args, models):
    (fold, train_index, test_index, X, y, dataset_name, strategy) = args
    all_model_results = []
    try:
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        train_df = pd.DataFrame(X_train_outer)
        train_df['target'] = y_train_outer

        params = {}
        if strategy in ["SMT", "GN"]:
            params = {'C.perc': 'balance', 'k': 5, 'pert': 0.1}
        elif strategy in ["RO", "RU"]:
            params = {'C.perc': 'balance'}
        elif strategy == "SG":
            params = {'samp_method': 'balance', 'k': 5, 'pert': 0.1}
        elif strategy == "WC":
            params = {'over': 0.5, 'under': 0.5}

        train_resampled = balance(train_df, strategy, params)
        if train_resampled is None:
            log_print(f"{RED}Resampling strategy {strategy} failed in fold {fold}, skipping it.{ENDC}", level=logging.ERROR)
            return all_model_results

        X_resampled = train_resampled.drop(columns='target').values
        y_resampled = train_resampled['target'].values

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_resampled)
        X_test_scaled = scaler.transform(X_test_outer)

        for model_name, model_info in models.items():
            log_print(f"Processing model: {model_name}, Fold: {fold}, Resampling: {strategy}")
            try:
                def objective(trial):
                    inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
                    param_space_model = model_info['params']
                    params_model = {}
                    for param_name, param_range in param_space_model.items():
                        if isinstance(param_range, list):
                            params_model[param_name] = trial.suggest_categorical(param_name, param_range)
                        elif isinstance(param_range, tuple) and all(isinstance(i, int) for i in param_range):
                            params_model[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        elif isinstance(param_range, tuple) and all(isinstance(i, float) for i in param_range):
                            params_model[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

                    model = model_info['model'](**params_model)
                    inner_cv = optuna.samplers.TPESampler(seed=42)
                    cv_scores = []

                    for inner_train_index, inner_val_index in inner_cv.split(X_train_scaled):
                        if time.time() - start_time > max_time:
                            log_print(f"{RED}Trial timed out, skipping current trial{ENDC}", level=logging.WARNING)
                            raise optuna.exceptions.TrialPruned()

                        X_train_inner, X_val_inner = X_train_scaled[inner_train_index], X_train_scaled[inner_val_index]
                        y_train_inner, y_val_inner = y_resampled[inner_train_index], y_resampled[inner_val_index]

                        try:
                            model.fit(X_train_inner, y_train_inner)
                        except Exception as e:
                            import traceback
                            log_print(f"{RED}Model training failed, error: {e}{ENDC}", level=logging.ERROR)
                            log_print(traceback.format_exc(), level=logging.ERROR)
                            raise optuna.exceptions.TrialPruned()

                        y_pred = model.predict(X_val_inner)
                        r2 = r2_score(y_val_inner, y_pred)
                        cv_scores.append(r2)

                    return -np.mean(cv_scores)

                sampler = TPESampler(seed=42)
                pruner = MedianPruner()
                study = optuna.create_study(
                    direction='minimize',
                    sampler=sampler,
                    pruner=pruner,
                    study_name=f"{model_name}_fold_{fold}_{strategy}",
                    storage='sqlite:///optuna_study.db',
                    load_if_exists=True
                )
                study.optimize(objective, n_trials=50, timeout=3600)

                best_params_model = study.best_params
                best_r2 = -study.best_value
                log_print(f"Fold {fold} Model {model_name} Best params: {study.best_params}, Best R²: {best_r2}, Resampling: {strategy}")

                model_final = model_info['model'](**best_params_model)
                try:
                    model_final.fit(X_train_scaled, y_resampled)
                except Exception as e:
                    import traceback
                    log_print(f"{RED}External model training failed, error: {e}{ENDC}", level=logging.ERROR)
                    log_print(traceback.format_exc(), level=logging.ERROR)
                    result = {
                        'Fold': f"{fold}",
                        'Model': model_name,
                        'Resampling': strategy,
                        'BestParams': study.best_params,
                        'RMSE': np.nan,
                        'MAE': np.nan,
                        'R2': np.nan,
                        'Pearson_R': np.nan,
                        'Status': 'Model Training Failed'
                    }
                    all_model_results.append(result)
                    continue

                try:
                    y_pred_outer = model_final.predict(X_test_scaled)
                except Exception as e:
                    import traceback
                    log_print(f"{RED}External prediction failed, error: {e}{ENDC}", level=logging.ERROR)
                    log_print(traceback.format_exc(), level=logging.ERROR)
                    result = {
                        'Fold': f"{fold}",
                        'Model': model_name,
                        'Resampling': strategy,
                        'BestParams': study.best_params,
                        'RMSE': np.nan,
                        'MAE': np.nan,
                        'R2': np.nan,
                        'Pearson_R': np.nan,
                        'Status': 'Prediction Failed'
                    }
                    all_model_results.append(result)
                    continue

                try:
                    rmse = np.sqrt(mean_squared_error(y_test_outer, y_pred_outer))
                    mae = mean_absolute_error(y_test_outer, y_pred_outer)
                    evs = explained_variance_score(y_test_outer, y_pred_outer)
                    r2 = r2_score(y_test_outer, y_pred_outer)
                    r, _ = pearsonr(y_test_outer, y_pred_outer)
                except Exception as e:
                    log_print(f"{RED}Calculating evaluation metrics failed, error: {e}{ENDC}", level=logging.ERROR)
                    rmse = mae = evs = r2 = r = np.nan

                result = {
                    'Fold': f"{fold}",
                    'Model': model_name,
                    'Resampling': strategy,
                    'BestParams': study.best_params,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Pearson_R': r,
                    'Status': 'Success'
                }
                all_model_results.append(result)
            except Exception as exc:
                log_print(f"{RED}Model {model_name} Fold {fold} Exception: {exc}{ENDC}", level=logging.ERROR)
                result = {
                    'Fold': f"{fold}",
                    'Model': model_name,
                    'Resampling': strategy,
                    'BestParams': None,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'R2': np.nan,
                    'Pearson_R': np.nan,
                    'Status': 'Exception Occurred'
                }
                all_model_results.append(result)
    except Exception as exc:
        log_print(f"{RED}Fold {fold} Exception: {exc}{ENDC}", level=logging.ERROR)
        result = {
            'Fold': f"{fold}",
            'Model': None,
            'Resampling': strategy,
            'BestParams': None,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Pearson_R': np.nan,
            'Status': 'Exception Occurred'
        }
        all_model_results.append(result)

    return all_model_results

def main():
    log_print("Starting the main function...")

    try:
        X_train_original, y_train_original = read_data('train_file_path')
        log_print("Training data loaded.")
    except Exception as e:
        log_print(f"Failed to load        training data, error: {e}", level=logging.ERROR)
        return

    try:
        X_test_original, y_test_original = read_data('test_file_path')
        log_print("Testing data loaded.")
    except Exception as e:
        log_print(f"Failed to load testing data, error: {e}", level=logging.ERROR)
        return

    results = []
    num_runs = 1
    resampling_strategies = ["SMT", "RO", "RU", "GN", "SG", "WC"]

    # Define machine learning models and their hyperparameter search spaces
    models = {
        'RandomForest': {'model': RandomForestRegressor, 'params': {'n_estimators': (50, 200), 'max_depth': (5, 20),
                                                                   'min_samples_split': (2, 10),
                                                                   'min_samples_leaf': (1, 10),
                                                                   'max_features': ['auto', 'sqrt', 'log2']}},
        'ExtraTrees': {'model': ExtraTreesRegressor, 'params': {'n_estimators': (50, 200), 'max_depth': (5, 20),
                                                                'min_samples_split': (2, 10),
                                                                'min_samples_leaf': (1, 10),
                                                                'max_features': ['auto', 'sqrt', 'log2']}},
        'AdaBoost': {'model': AdaBoostRegressor, 'params': {'n_estimators': (50, 200), 'learning_rate': (0.01, 1.0),
                                                            'loss': ['linear', 'square', 'exponential']}},
        'GradientBoosting': {'model': GradientBoostingRegressor,
                             'params': {'n_estimators': (50, 200), 'learning_rate': (0.01, 1.0),
                                        'max_depth': (3, 10), 'min_samples_split': (2, 10),
                                        'min_samples_leaf': (1, 10), 'subsample': (0.5, 1.0),
                                        'max_features': ['auto', 'sqrt', 'log2']}},
        'XGBoost': {'model': xgb.XGBRegressor,
                    'params': {'n_estimators': (50, 200), 'learning_rate': (0.01, 0.3),
                               'max_depth': (3, 10), 'subsample': (0.5, 1.0),
                               'colsample_bytree': (0.5, 1.0), 'gamma': (0, 5),
                               'reg_alpha': (0, 1), 'reg_lambda': (0, 1)}},
        'LightGBM': {'model': LGBMRegressor,
                     'params': {'n_estimators': (50, 200), 'learning_rate': (0.01, 0.3),
                                'max_depth': (-1, 15), 'num_leaves': (20, 150),
                                'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0),
                                'reg_alpha': (0, 1), 'reg_lambda': (0, 1)}},
        'SVR': {'model': SVR, 'params': {'C': (0.1, 1000), 'epsilon': (0.01, 1.0), 'gamma': ['scale', 'auto']}},
        'Ridge': {'model': Ridge, 'params': {'alpha': (0.1, 100.0), 'fit_intercept': [True, False],
                                             'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']}},
        'LinearRegression': {'model': LinearRegression, 'params': {'fit_intercept': [True, False]}},
        'Lasso': {'model': Lasso, 'params': {'alpha': (0.001, 1.0), 'fit_intercept': [True, False],
                                             'max_iter': (100, 1000)}},
        'DecisionTree': {'model': DecisionTreeRegressor,
                         'params': {'max_depth': (3, 20), 'min_samples_split': (2, 10),
                                    'min_samples_leaf': (1, 10), 'max_features': ['auto', 'sqrt', 'log2']}},
        'KNeighbors': {'model': KNeighborsRegressor,
                       'params': {'n_neighbors': (1, 30), 'weights': ['uniform', 'distance'],
                                  'metric': ['euclidean', 'manhattan', 'minkowski']}},
        'Bagging': {'model': BaggingRegressor,
                    'params': {'n_estimators': (10, 100), 'max_samples': (0.5, 1.0),
                               'max_features': (0.5, 1.0), 'bootstrap': [True, False]}}
    }

    for i in range(num_runs):
        print(f"\nStarting iteration {i + 1}/{num_runs}")
        log_print(f"\nStarting iteration {i + 1}/{num_runs}")

        random_state = np.random.randint(0, 10000)

        X_train, y_train = X_train_original.copy(), y_train_original.copy()
        X_test, y_test = X_test_original.copy(), y_test_original.copy()

        for strategy in resampling_strategies:
            log_print(f"\nApplying resampling strategy: {strategy}")
            print(f"\nApplying resampling strategy: {strategy}")

            tasks = []
            outer = KFold(n_splits=10, random_state=random_state, shuffle=True)
            for fold, (train_index, test_index) in enumerate(outer.split(X_train, y_train)):
                tasks.append((fold + 1, train_index, test_index, X_train, y_train, 'dataset_name', strategy))

            all_model_results = []
            max_workers = min(4, len(tasks) * len(models))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(process_fold, task, models): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        model_results = future.result()
                        if model_results is not None:
                            all_model_results.extend(model_results)
                        else:
                            fold, _, _, _, _, _, _ = task
                            log_print(f"{RED}Fold {fold} no results returned, skipped{ENDC}", level=logging.WARNING)
                    except Exception as exc:
                        fold, _, _, _, _, _, _ = task
                        log_print(f"{RED}Fold {fold} Exception: {exc}{ENDC}", level=logging.ERROR)

            df_results = pd.DataFrame(all_model_results)
            results.extend(all_model_results)

    df_final_results = pd.DataFrame(results)
    if 'BestParams' in df_final_results.columns:
        params_df = df_final_results['BestParams'].apply(pd.Series)
        df_final_results = pd.concat([df_final_results.drop('BestParams', axis=1), params_df], axis=1)

    try:
        df_final_results.to_excel('self_model+resampling_results.xlsx', index=False)
        log_print("\nResults saved to 'self_model+resampling_results.xlsx'.")
        print("\nResults saved to 'self_model+resampling_results.xlsx'.")
    except Exception as e:
        log_print(f"Failed to save results to Excel, error: {e}", level=logging.ERROR)
        print(f"Failed to save results to Excel, error: {e}")


def read_data(file_path):
    try:
        df = pd.read_excel(file_path)
        X = df.drop(columns=['target']).values
        y = df['target'].values.flatten()
        return X, y
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error reading data: {e}")
        raise


if __name__ == "__main__":
    main()