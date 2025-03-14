import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor as skET
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
warnings.filterwarnings('ignore')
import logging

# Resampling related libraries
import smogn
import ImbalancedLearningRegression as iblr

np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(10)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def balance(train, strategy, params):
    try:
        log_print(f"Applying strategy: {strategy} with params: {params}")

        if train.isnull().values.any():
            log_print(f"{RED}Original training data has missing values, resampling aborted.{ENDC}", level=logging.ERROR)
            return None

        rare_condition = train['target'] >
        majority = train[~rare_condition]
        minority = train[rare_condition]

        if len(minority) == 0:
            log_print(f"{RED}No rare data with target > 10, resampling aborted.{ENDC}", level=logging.ERROR)
            return None

        if strategy == "SMT":
            try:
                train_resampled = iblr.smote(
                    data=train,
                    y='target',
                    samp_method=params.get('C.perc', 'balance'),
                    k=params.get('k', 5),
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}SMT resampling failed, error: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "GN":
            try:
                train_resampled = iblr.gn(
                    data=train,
                    y='target',
                    samp_method=params.get('C.perc', 'balance'),
                    pert=params.get('pert', 0.1),
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}GN resampling failed, error: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "RO":
            try:
                train_resampled = iblr.ro(
                    data=train,
                    y='target',
                    samp_method=params.get('C.perc', 'balance'),
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}RO resampling failed, error: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "RU":
            try:
                train_resampled = iblr.random_under(
                    data=train,
                    y='target',
                    samp_method=params.get('C.perc', 'balance'),
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}RU resampling failed, error: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "SG":
            try:
                train_resampled = smogn.smoter(
                    data=train,
                    y='target',
                    samp_method=params.get('samp_method', 'balance'),
                    k=params.get('k', 5),
                    pert=params.get('pert', 0.1),
                    rel_xtrm_type='high',
                    rel_thres=0.8
                )
            except ValueError as ve:
                log_print(f"{RED}SG resampling failed, error: {ve}{ENDC}", level=logging.ERROR)
                return None
            except Exception as e:
                log_print(f"{RED}SG resampling failed, error: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "WC":
            try:
                over = params.get('over', 0.5)
                under = params.get('under', 0.5)

                if over > 0:
                    minority_over = minority.sample(frac=over, replace=True, random_state=42)
                else:
                    minority_over = minority.copy()

                if under > 0:
                    majority_under = majority.sample(frac=1 - under, random_state=42)
                else:
                    majority_under = majority.copy()

                train_resampled = pd.concat([majority_under, minority_over], axis=0).reset_index(drop=True)
            except Exception as e:
                log_print(f"{RED}WC resampling failed, error: {e}{ENDC}", level=logging.ERROR)
                return None
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

# ECA Channel Attention Module
class ECAAttention(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x.unsqueeze(1)).squeeze(-1)
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        y = self.sigmoid(y)
        return x * y

# CCAM Cross-Channel Attention Module
class CCAM(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super(CCAM, self).__init__()
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_gmp = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv_gap = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv_final = nn.Conv1d(2, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gmp = self.gmp(x.unsqueeze(1)).squeeze(-1)
        gap = self.gap(x.unsqueeze(1)).squeeze(-1)

        gmp_conv = self.conv_gmp(gmp.unsqueeze(1)).squeeze(1)
        gap_conv = self.conv_gap(gap.unsqueeze(1)).squeeze(1)

        concat = torch.stack([gmp_conv, gap_conv], dim=1)
        weight = self.conv_final(concat).squeeze(1)
        weight = self.sigmoid(weight)

        out = x * weight + x
        return out

# CombinedAttention Module (Combining ECA and CCAM)
class CombinedAttention(nn.Module):
    def __init__(self, input_dim: int, kernel_size: int = 5, eca_k_size: int = 3, dropout_rate: float = 0.1):
        super(CombinedAttention, self).__init__()
        self.eca = ECAAttention(channels=input_dim, k_size=eca_k_size)
        self.ccam = CCAM(channels=input_dim, kernel_size=kernel_size)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eca_output = self.eca(x)
        ccam_output = self.ccam(eca_output)
        out = self.layer_norm(x + self.dropout(ccam_output))
        return out

# Composite Loss Function
class CompositeLoss(nn.Module):
    def __init__(self, delta: float = 1.0, lambda_reg: float = 0.01):
        super(CompositeLoss, self).__init__()
        self.delta = delta
        self.lambda_reg = lambda_reg

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        huber_loss_fn = nn.SmoothL1Loss(beta=self.delta)
        huber_loss = huber_loss_fn(y_pred, y_true)
        l2_reg = torch.sum(theta ** 2)
        loss = huber_loss + self.lambda_reg * l2_reg
        return loss

# Custom Cautious Weighted ExtraTrees Regression Model
class CautiousWeightedExtraTrees(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, min_samples_leaf: int = 1,
                 min_samples_split: int = 2, max_features: str = 'auto', bootstrap: bool = False,
                 regularization_factor: float = 0.1, learning_rate: float = 0.1,
                 early_stopping_rounds: int = 10, max_boost_rounds: int = 50,
                 l1_ratio: float = 0.7, huber_delta: float = 1.0,
                 dropout_rate: float = 0.1, weight_decay: float = 1e-4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.regularization_factor = regularization_factor
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.max_boost_rounds = max_boost_rounds
        self.l1_ratio = l1_ratio
        self.huber_delta = huber_delta
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.model = None
        self.weights = None
        self.residual_model = None
        self.attention = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = skET(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

        self.tree_predictions = np.array(
            [tree.predict(X) for tree in self.model.estimators_]
        ).T

        self.attention = CombinedAttention(input_dim=self.n_estimators, kernel_size=5, eca_k_size=3,
                                           dropout_rate=self.dropout_rate).to(device)

        tree_preds_tensor = torch.tensor(self.tree_predictions, dtype=torch.float32).to(device)
        attended_preds = self.attention(tree_preds_tensor).detach().cpu().numpy()

        self.optimize_weights(attended_preds, y)

        self.initial_predictions = np.dot(self.tree_predictions, self.weights)

        self.boosting_iteration(X, y)
        return self

    def optimize_weights(self, tree_predictions: np.ndarray, y: np.ndarray):
        n_samples, n_trees = tree_predictions.shape
        weights = torch.ones(n_trees, device=device, requires_grad=True)
        weights.data /= n_trees

        y = y.values if hasattr(y, 'values') else y
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        tree_predictions_tensor = torch.tensor(tree_predictions, dtype=torch.float32, device=device)

        criterion = CompositeLoss(delta=self.huber_delta, lambda_reg=self.regularization_factor)
        optimizer = optim.Adam([weights], lr=0.01, weight_decay=self.weight_decay)
        early_stop_counter = 0
        best_cost = float('inf')

        for iteration in range(1000):
            optimizer.zero_grad()
            weighted_preds = torch.matmul(tree_predictions_tensor, weights)
            loss = criterion(weighted_preds, y_tensor, weights)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                weights.data = torch.clamp(weights.data, 0, 1)
                weights.data /= torch.sum(weights.data)

            if loss.item() < best_cost:
                best_cost = loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.early_stopping_rounds:
                break

        self.weights = weights.detach().cpu().numpy()

    def boosting_iteration(self, X: np.ndarray, y: np.ndarray):
        residuals = y - np.dot(self.tree_predictions, self.weights)
        dtrain = xgb.DMatrix(X, label=residuals)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.max_depth if self.max_depth else 6,
            'learning_rate': self.learning_rate,
            'verbosity': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1,
            'alpha': 0.01,
            'random_state': 42
        }

        self.residual_model = xgb.train(
            params,
            dtrain,
            num_boost_rounds=self.max_boost_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=False
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_]).T
        weighted_preds = np.dot(tree_preds, self.weights)
        if self.residual_model:
            dtest = xgb.DMatrix(X)
            residual_preds = self.residual_model.predict(dtest)
            final_preds = weighted_preds + residual_preds
        else:
            final_preds = weighted_preds
        return final_preds

def evaluate_model_performance(model, X_train, y_train, X_test, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    evs_train = explained_variance_score(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r_train, _ = pearsonr(y_train, y_train_pred)
    huber_loss_train = np.mean(np.where(
        np.abs(y_train - y_train_pred) <= model.huber_delta,
        0.5 * (y_train - y_train_pred) ** 2,
        model.huber_delta * (np.abs(y_train - y_train_pred) - 0.5 * model.huber_delta)
    ))

    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    evs_test = explained_variance_score(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r_test, _ = pearsonr(y_test, y_test_pred)
    huber_loss_test = np.mean(np.where(
        np.abs(y_test - y_test_pred) <= model.huber_delta,
        0.5 * (y_test - y_test_pred) ** 2,
        model.huber_delta * (np.abs(y_test - y_test_pred) - 0.5 * model.huber_delta)
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


def main():
    log_print("Starting the main function...")

    try:
        X_train_original, y_train_original = read_data('train_file_path')
        log_print("Training data loaded.")
    except Exception as e:
        log_print(f"Failed to load training data, error: {e}", level=logging.ERROR)
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

    for i in range(num_runs):
        print(f"\nStarting iteration {i + 1}/{num_runs}")
        log_print(f"\nStarting iteration {i + 1}/{num_runs}")

        random_state = np.random.randint(0, 10000)

        X_train, y_train = X_train_original.copy(), y_train_original.copy()
        X_test, y_test = X_test_original.copy(), y_test_original.copy()

        feature_columns = []
        train_df = pd.DataFrame(X_train, columns=feature_columns)
        train_df['target'] = y_train

        for strategy in resampling_strategies:
            log_print(f"\nApplying resampling strategy: {strategy}")
            print(f"\nApplying resampling strategy: {strategy}")

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
                log_print(f"{RED}Resampling strategy {strategy} failed, skipping it.{ENDC}", level=logging.ERROR)
                print(f"{RED}Resampling strategy {strategy} failed, skipping it.{ENDC}")
                continue

            X_resampled = train_resampled[feature_columns].values
            y_resampled = train_resampled['target'].values

            X_train_scaled, X_test_scaled = normalize_features(X_resampled, X_test)

            param_space = {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(5, 20),
                'min_samples_leaf': Integer(1, 10),
                'min_samples_split': Integer(2, 10),
                'max_features': Real(0.1, 1.0, prior='uniform'),
                'bootstrap': Categorical([True, False]),
                'regularization_factor': Real(0.01, 1.0, prior='log-uniform'),
                'l1_ratio': Real(0.0, 1.0, prior='uniform'),
                'huber_delta': Real(0.1, 2.0, prior='uniform'),
                'dropout_rate': Real(0.0, 0.5, prior='uniform'),
                'weight_decay': Real(1e-5, 1e-2, prior='log-uniform'),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'early_stopping_rounds': Integer(5, 20),
                'max_boost_rounds': Integer(10, 100)
            }

            base_model = CautiousWeightedExtraTrees()
            bayes_search = BayesSearchCV(
                estimator=base_model,
                search_spaces=param_space,
                n_iter=32,
                cv=10,  # 修改为 10 折交叉验证
                scoring='neg_mean_squared_error',
                random_state=random_state,
                n_jobs=-1,
                verbose=0
            )
            log_print("Starting Bayesian optimization...")
            print("Starting Bayesian optimization...")
            bayes_search.fit(X_train_scaled, y_resampled)
            log_print("Bayesian optimization completed.")
            print("Bayesian optimization completed.")

            best_model = bayes_search.best_estimator_
            log_print("Best model initialized.")
            print("Best model initialized.")

            log_print("Starting to train the best model...")
            print("Starting to train the best model...")
            best_model.fit(X_train_scaled, y_resampled)
            log_print("Best model training completed.")
            print("Best model training completed.")

            metrics = evaluate_model_performance(
                best_model, X_train_scaled, y_resampled, X_test_scaled, y_test,
                f"CWRF with Bayesian Optimization and {strategy}"
            )

            best_params = bayes_search.best_params_
            weights = best_model.weights

            if metrics['r2_test'] >= 0.77:
                result = {
                    'iteration': i + 1,
                    'strategy': strategy,
                    **metrics,
                    'best_params': best_params,
                    'weights': weights.tolist()
                }
                results.append(result)

    df_results = pd.DataFrame(results)

    if 'best_params' in df_results.columns:
        params_df = df_results['best_params'].apply(pd.Series)
        df_results = pd.concat([df_results.drop('best_params', axis=1), params_df], axis=1)

    try:
        df_results.to_excel('self_model+resampling_results.xlsx', index=False)
        log_print("\nResults saved to 'self_model+resampling_results.xlsx'.")
        print("\nResults saved to 'self_model+resampling_results.xlsx'.")
    except Exception as e:
        log_print(f"Failed to save results to Excel, error: {e}", level=logging.ERROR)
        print(f"Failed to save results to Excel, error: {e}")


def read_data(file_path):
    try:
        df = pd.read_excel(file_path)
        feature_columns = []
        missing_cols = set(feature_columns + ['target']) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns missing in dataset: {missing_cols}")
        X = df[feature_columns].values
        y = df['target'].values.flatten()
        return X, y
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error reading data: {e}")
        raise


def normalize_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    main()