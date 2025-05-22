import random
import numpy as np
import pandas as pd
import logging
import warnings
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import smogn
import ImbalancedLearningRegression as iblr  # 请确保已正确安装该库
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import torch
import xgboost as xgb
import os
from sklearn.base import BaseEstimator, RegressorMixin
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import ExtraTreesRegressor
import torch.nn.functional as F
import multiprocessing
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import MultipleLocator

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_Dmax_histogram(y):
    """绘制Dmax的直方图（原始功能）"""
    plt.figure(figsize=(8, 6))
    plt.hist(y, 70, color='red', edgecolor='black')
    plt.xlabel('Dmax')
    plt.ylabel('频数')
    plt.title('Dmax分布直方图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "dmax_histogram.png"))
    plt.close()

def plot_data_distribution(y):
    """绘制目标变量Dmax的分布直方图"""
    sns.set_style("white")
    plt.figure(figsize=(8, 6))
    sns.distplot(y, color="b")
    plt.xlabel("Dmax", weight='bold')
    plt.ylabel("频率", weight='bold')
    plt.title("Dmax分布", weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "dmax_distribution.png"))
    plt.close()

def plot_feature_correlation(df, feature_columns):
    """绘制特征与Dmax的相关性散点图"""
    if not feature_columns:
        return

    plt.figure(figsize=(12, 8))
    sns.set_color_codes(palette='deep')

    for i, feature in enumerate(feature_columns):
        plt.subplot(2, len(feature_columns) // 2 + 1, i + 1)
        sns.scatterplot(x=feature, y='Dmax', data=df, palette='Blues')
        sns.regplot(x=feature, y='Dmax', data=df, scatter=False)
        plt.xlabel(feature, weight='bold')
        plt.ylabel('Dmax', weight='bold')
        plt.xticks(weight='bold')
        plt.yticks(weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "feature_correlation.png"))
    plt.close()

def plot_correlation_matrix(df, feature_columns):
    """绘制特征相关性热力图"""
    if not feature_columns:
        return

    # 构建特征相关矩阵
    corr_matrix = df[feature_columns + ['Dmax']].corr()

    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, center=0,
                annot=True, square=True, linewidths=.5, fmt='.2f')
    plt.title("特征相关性热力图", weight='bold')
    plt.xticks(weight='bold', rotation=45)
    plt.yticks(weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "correlation_matrix.png"))
    plt.close()

# 设置基本路径
base_path = r""
# 设置随机种子
seed = 42  # 设置一个固定的随机种子值
# 为 numpy 设置种子
np.random.seed(seed)
# 为 random 设置种子（Python 内置的 random 库）
random.seed(seed)
# 为 PyTorch 设置种子
torch.manual_seed(seed)
# 如果你使用 GPU，确保 CUDA 的种子设置正确
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 定义日志配置
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# 定义日志打印函数
def log_print(message, level=logging.INFO, color=None):
    """
    打印消息并记录到日志文件中。

    参数:
        message (str): 要打印和记录的消息。
        level (int): 日志级别，默认为 logging.INFO。
        color (str, optional): ANSI 颜色代码，如 RED。默认为 None。
    """
    if color:
        message = f"{color}{message}\033[0m"
    logging.log(level, message)

# ANSI 转义码用于在终端中显示红色文字（可选）
RED = "\033[91m"
ENDC = "\033[0m"

# 忽略警告
warnings.filterwarnings('ignore')

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义模型定义
class SelfAttention(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.input_dim)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, values)
        return output

class ECAAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x.unsqueeze(1)).squeeze(-1)
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        y = self.sigmoid(y)
        return x * y

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

class ResidualSelfAttentionWithECAandDAE(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1, eca_k_size=3):
        super(ResidualSelfAttentionWithECAandDAE, self).__init__()
        self.attention = SelfAttention(input_dim, dropout_rate)
        self.eca = ECAAttention(input_dim, k_size=eca_k_size)
        self.dae = DenoisingAutoencoder(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        attn_output = self.attention(x)
        attn_output = self.eca(attn_output)
        denoised_output = self.dae(attn_output)
        out = self.layer_norm(x + self.dropout(denoised_output))
        return out

class CompositeLoss(nn.Module):
    def __init__(self, delta=1.0, lambda_reg=0.01):
        super(CompositeLoss, self).__init__()
        self.delta = delta
        self.lambda_reg = lambda_reg

    def forward(self, y_pred, y_true, theta):
        huber_loss_fn = nn.SmoothL1Loss(beta=self.delta)
        huber_loss = huber_loss_fn(y_pred, y_true)
        l2_reg = torch.sum(theta ** 2)
        loss = huber_loss + self.lambda_reg * l2_reg
        return loss

class EAGET(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                 max_features='auto', bootstrap=False, regularization_factor=0.1,
                 learning_rate=0.1, early_stopping_rounds=10, max_boost_rounds=50,
                 l1_ratio=0.7, huber_delta=1.0, dropout_rate=0.1, weight_decay=1e-4):
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
        self.denoise = None

    def fit(self, X, y):
        # 训练 ExtraTrees 模型
        self.model = ExtraTreesRegressor(
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

        # 获取每棵树的预测
        self.tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_]).T  # (n_samples, n_estimators)

        # 初始化带残差连接和ECA的自注意力机制
        self.attention = ResidualSelfAttentionWithECAandDAE(input_dim=self.n_estimators,
                                                            dropout_rate=self.dropout_rate,
                                                            eca_k_size=3).to(device)

        # 自注意力处理
        tree_preds_tensor = torch.tensor(self.tree_predictions, dtype=torch.float32).to(device)  # (n_samples, n_estimators)
        attended_preds = self.attention(tree_preds_tensor).detach().cpu().numpy()  # (n_samples, n_estimators)

        # 优化权重
        self.optimize_weights(attended_preds, y)

        # 保存第一轮的最终预测值（不包含XGBoost残差）
        self.initial_predictions = np.dot(self.tree_predictions, self.weights)

        # 残差拟合
        self.boosting_iteration(X, y)
        return self

    def optimize_weights(self, tree_predictions, y):
        n_samples, n_trees = tree_predictions.shape
        weights = torch.ones(n_trees, device=device, requires_grad=True)
        weights.data /= n_trees  # 初始化权重均匀分布

        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        tree_predictions_tensor = torch.tensor(tree_predictions, dtype=torch.float32, device=device)

        criterion = CompositeLoss(delta=self.huber_delta, lambda_reg=self.regularization_factor)
        optimizer = optim.Adam([weights], lr=0.01, weight_decay=self.weight_decay)
        early_stop_counter = 0
        best_cost = float('inf')

        for iteration in range(1000):
            optimizer.zero_grad()
            weighted_preds = torch.matmul(tree_predictions_tensor, weights)  # (n_samples,)
            loss = criterion(weighted_preds, y_tensor, weights)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                weights.data = torch.clamp(weights.data, 0, 1)  # 保证权重在0和1之间
                weights.data /= torch.sum(weights.data)  # 归一化权重

            if loss.item() < best_cost:
                best_cost = loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.early_stopping_rounds:
                break

        self.weights = weights.detach().cpu().numpy()

    def boosting_iteration(self, X, y):
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
        }

        self.residual_model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.max_boost_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            evals=[(dtrain, "train")],
            verbose_eval=False
        )

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_]).T  # (n_samples, n_estimators)
        weighted_preds = np.dot(tree_preds, self.weights)  # (n_samples,)
        if self.residual_model:
            dtest = xgb.DMatrix(X)
            residual_preds = self.residual_model.predict(dtest)  # (n_samples,)
            final_preds = weighted_preds + residual_preds
        else:
            final_preds = weighted_preds
        return final_preds

def balance(train, strategy, params):
    """
    重采样函数，根据策略和参数对训练数据进行重采样。

    参数:
        train (pd.DataFrame): 训练数据集，包含目标变量。
        strategy (str): 重采样策略名称。
        params (dict): 重采样参数。

    返回:
        pd.DataFrame 或 None: 重采样后的训练数据集，如果失败则返回 None。
    """
    try:
        log_print(f"开始应用策略: {strategy}，参数: {params}", level=logging.INFO)

        # 提前检查数据是否存在缺失值
        if train.isnull().values.any():
            log_print(f"{RED}原始训练数据存在缺失值，无法进行重采样。{ENDC}", level=logging.ERROR)
            return None

        # 根据 Dmax > 10 定义稀有数据
        rare_condition = train['Dmax'] >
        majority = train[~rare_condition]
        minority = train[rare_condition]

        if len(minority) == 0:
            log_print(f"{RED}没有满足 Dmax >  的稀有数据，无法进行重采样。{ENDC}", level=logging.ERROR)
            return None

        # 根据不同的策略应用重采样方法
        if strategy == "SMT":
            try:
                train_resampled = iblr.smote(
                    data=train,
                    y='Dmax',
                    samp_method=params['C.perc'],
                    k=params['k'],
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}SMT 重采样失败，错误: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "GN":
            try:
                train_resampled = iblr.gn(
                    data=train,
                    y='Dmax',
                    samp_method=params['C.perc'],
                    pert=params['pert'],
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}GN 重采样失败，错误: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "RO":
            try:
                train_resampled = iblr.ro(
                    data=train,
                    y='Dmax',
                    samp_method=params['C.perc'],
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}RO 重采样失败，错误: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "RU":
            try:
                train_resampled = iblr.random_under(
                    data=train,
                    y='Dmax',
                    samp_method=params['C.perc'],
                    rel_thres=0.8
                )
            except Exception as e:
                log_print(f"{RED}RU 重采样失败，错误: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "SG":
            try:
                train_resampled = smogn.smoter(
                    data=train,
                    y='Dmax',
                    samp_method=params['samp_method'],
                    k=params['k'],
                    pert=params['pert'],
                    rel_xtrm_type='high',
                    rel_thres=0.8
                )
            except ValueError as ve:
                log_print(f"{RED}SG 重采样失败，错误: {ve}{ENDC}", level=logging.ERROR)
                return None
            except Exception as e:
                log_print(f"{RED}SG 重采样失败，错误: {e}{ENDC}", level=logging.ERROR)
                return None
        elif strategy == "WC":
            try:
                C_perc = params['C.perc']
                over = params['over']
                under = params['under']
                # 根据 C.perc 调整过采样/欠采样强度
                if C_perc == 'extreme':
                    over = min(over + 0.2, 0.7)  # 极端模式下增强过采样比例
                    under = max(under - 0.2, 0.3)  # 极端模式下增强欠采样比例

                # 过采样稀有数据
                if over > 0:
                    minority_over = minority.sample(frac=over, replace=True, random_state=42)
                else:
                    minority_over = minority.copy()

                # 欠采样多数数据
                if under > 0:
                    majority_under = majority.sample(frac=1 - under, random_state=42)
                else:
                    majority_under = majority.copy()

                train_resampled = pd.concat([majority_under, minority_over], axis=0).reset_index(drop=True)
            except Exception as e:
                log_print(f"{RED}WC 重采样失败，错误: {e}{ENDC}", level=logging.ERROR)
                return None
        else:
            log_print(f"{RED}未知的重采样策略: {strategy}{ENDC}", level=logging.ERROR)
            return None        # 检查重采样后的数据是否有缺失值
        if train_resampled.isnull().values.any():
            log_print(f"{RED}策略 {strategy} 重采样后存在缺失值，尝试删除缺失值{ENDC}", level=logging.WARNING)
            train_resampled = train_resampled.dropna()
            if train_resampled.empty:
                log_print(f"{RED}重采样后数据为空，策略: {strategy}, 参数: {params}{ENDC}", level=logging.ERROR)
                return None

        # 检查重采样后的数据是否只有一个类别
        y_resampled_values = train_resampled['Dmax'].unique()
        if len(y_resampled_values) < 2:
            log_print(f"{RED}重采样后目标变量只有一个唯一值，策略: {strategy}{ENDC}", level=logging.ERROR)
            return None

        return train_resampled
    except Exception as e:
        log_print(f"{RED}平衡函数发生异常，错误: {e}{ENDC}", level=logging.ERROR)
        return None


def process_fold_strategy(args, strategys, models, best_r2_so_far, best_train_test_data):
    """
    处理单个策略和模型在单个折叠（fold）中的优化和评估。

    参数:
        args (tuple): 包含策略名称、折叠编号、训练索引、测试索引、X、y、dataset_name 等信息。
        strategys (dict): 重采样策略及其参数范围。
        models (dict): 模型及其超参数范围。
    返回:
        list of dict: 评估结果。
    """
    (strategy, fold, train_index, test_index, X, y, dataset_name) = args
    all_model_results = []
    try:
        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        # 构建训练 DataFrame
        feature_columns = []
        train_df = pd.DataFrame(X_train_outer, columns=feature_columns)
        train_df.insert(0, 'Dmax', y_train_outer)

        for model_name, model_info in models.items():
            log_print(f"开始处理策略: {strategy}，模型: {model_name}，Fold: {fold}", level=logging.INFO)
            try:
                def objective(trial, cv_rmse=None):
                    start_time = time.time()
                    max_time = 300  # 设置每个 trial 的最大执行时间为 300 秒

                    # 获取重采样策略的超参数范围
                    param_space_resampling = strategys[strategy]

                    # 根据策略定义参数搜索空间
                    params_resampling = {}
                    for param_name, param_range in param_space_resampling.items():
                        if isinstance(param_range, list):
                            params_resampling[param_name] = trial.suggest_categorical(param_name, param_range)
                        elif isinstance(param_range, tuple) and all(isinstance(i, int) for i in param_range):
                            params_resampling[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        elif isinstance(param_range, tuple) and all(isinstance(i, float) for i in param_range):
                            params_resampling[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

                    # 应用重采样策略
                    train_resampled = balance(train_df, strategy, params_resampling)
                    if train_resampled is None or train_resampled.empty:
                        raise optuna.exceptions.TrialPruned()

                    X_resampled = train_resampled.drop(["Dmax"], axis=1).to_numpy()
                    y_resampled = train_resampled["Dmax"].to_numpy()

                    # 特征归一化，有助于防止过拟合
                    scaler = MinMaxScaler()
                    X_resampled = scaler.fit_transform(X_resampled)

                    # 获取模型的超参数搜索空间
                    param_space_model = model_info['params']
                    params_model = {}
                    for param_name, param_range in param_space_model.items():
                        if isinstance(param_range, list):
                            params_model[param_name] = trial.suggest_categorical(param_name, param_range)
                        elif isinstance(param_range, tuple) and all(isinstance(i, int) for i in param_range):
                            params_model[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        elif isinstance(param_range, tuple) and all(isinstance(i, float) for i in param_range):
                            params_model[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

                    # 初始化模型
                    model = model_info['model'](**params_model)

                    # 使用交叉验证评估模型性能，防止过拟合
                    inner_cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
                    cv_rmse = []

                    for inner_train_index, inner_val_index in inner_cv.split(X_resampled):
                        if time.time() - start_time > max_time:
                            log_print(f"{RED}Trial 超时，跳过当前 trial{ENDC}", level=logging.WARNING)
                            raise optuna.exceptions.TrialPruned()

                        X_train_inner, X_val_inner = X_resampled[inner_train_index], X_resampled[inner_val_index]
                        y_train_inner, y_val_inner = y_resampled[inner_train_index], y_resampled[inner_val_index]

                        # 训练模型
                        try:
                            model.fit(X_train_inner, y_train_inner)
                        except Exception as e:
                            import traceback
                            log_print(f"{RED}模型训练失败，错误: {e}{ENDC}", level=logging.ERROR)
                            log_print(traceback.format_exc(), level=logging.ERROR)
                            raise optuna.exceptions.TrialPruned()

                        # 预测
                        y_pred = model.predict(X_val_inner)
                        # 计算 RMSE
                        rmse = np.sqrt(mean_squared_error(y_val_inner, y_pred))
                        cv_rmse.append(rmse)

                    # 返回平均 RMSE（Optuna 会最小化此值）
                    return np.mean(cv_rmse)

                # 创建 Optuna 学习器，使用 TPE 采样器和 MedianPruner 剪枝器
                sampler = TPESampler(seed=42)
                pruner = MedianPruner()
                study = optuna.create_study(
                    direction='minimize',
                    sampler=sampler,
                    pruner=pruner
                )
                study.optimize(objective, n_trials=50, timeout=3600)  # 根据需求调整试验次数和超时时间

                # 获取并应用最佳超参数（重采样和模型）
                best_params_resampling = {k: v for k, v in study.best_params.items() if k in strategys[strategy]}
                best_params_model = {k: v for k, v in study.best_params.items() if
                                     k not in strategys[strategy] and not k.startswith('hidden_layer_sizes_layer_') and k != 'hidden_layer_sizes_n_layers'}
                if any(k.startswith('hidden_layer_sizes_layer_') for k in study.best_params):
                    hidden_layers = [v for k, v in study.best_params.items() if k.startswith('hidden_layer_sizes_layer_')]
                    best_params_model['hidden_layer_sizes'] = tuple(hidden_layers)

                # 使用最佳参数在外部训练集上训练模型
                train_resampled = balance(train_df, strategy, best_params_resampling)
                if train_resampled is None or train_resampled.empty:
                    log_print(f"重采样失败，策略: {strategy}, 参数: {best_params_resampling}", level=logging.ERROR)
                    continue

                X_resampled = train_resampled.drop(["Dmax"], axis=1).to_numpy()
                y_resampled = train_resampled["Dmax"].to_numpy()

                # 特征归一化
                scaler = MinMaxScaler()
                X_resampled = scaler.fit_transform(X_resampled)
                X_test_outer_scaled = scaler.transform(X_test_outer)

                # 初始化模型
                model_final = model_info['model'](**best_params_model)

                # 训练最终模型
                try:
                    model_final.fit(X_resampled, y_resampled)
                except Exception as e:
                    import traceback
                    log_print(f"{RED}外部模型训练失败，错误: {e}{ENDC}", level=logging.ERROR)
                    log_print(traceback.format_exc(), level=logging.ERROR)
                    continue

                # 预测外部测试集
                try:
                    y_pred_outer = model_final.predict(X_test_outer_scaled)
                except Exception as e:
                    import traceback
                    log_print(f"{RED}外部预测失败，错误: {e}{ENDC}", level=logging.ERROR)
                    log_print(traceback.format_exc(), level=logging.ERROR)
                    continue

                # 计算评估指标
                try:
                    rmse = np.sqrt(mean_squared_error(y_test_outer, y_pred_outer))
                    mae = mean_absolute_error(y_test_outer, y_pred_outer)
                    evs = explained_variance_score(y_test_outer, y_pred_outer)
                    r2 = r2_score(y_test_outer, y_pred_outer)
                    r, _ = pearsonr(y_test_outer, y_pred_outer)
                except Exception as e:
                    log_print(f"{RED}计算评估指标失败，错误: {e}{ENDC}", level=logging.ERROR)
                    rmse = mae = evs = r2 = r = np.nan

                # 保存平均评估结果
                result = {
                    'Fold': f"{fold}",
                    'Strategy': strategy,
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'Pearson_R': r,
                    'Status': 'Success'
                }
                all_model_results.append(result)

            except Exception as exc:
                log_print(f"{RED}策略 {strategy} 模型 {model_name} Fold {fold} 生成异常: {exc}{ENDC}", level=logging.ERROR)
                result = {
                    'Fold': f"{fold}",
                    'Strategy': strategy,
                    'Model': model_name,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'R2': np.nan,
                    'Pearson_R': np.nan,
                    'Status': 'Exception Occurred'
                }
                all_model_results.append(result)

    except Exception as exc:
        log_print(f"{RED}策略 {strategy} Fold {fold} 生成异常: {exc}{ENDC}", level=logging.ERROR)
        result = {
            'Fold': f"{fold}",
            'Strategy': strategy,
            'Model': None,
            'RMSE': np.nan,
            'MAE': np.nan,
            'R2': np.nan,
            'Pearson_R': np.nan,
            'Status': 'Exception Occurred'
        }
        all_model_results.append(result)

    return all_model_results


def repeatedKfold(X, y, dataset_name, X_test_final, y_test_final, models):
    """
    使用嵌套交叉验证和贝叶斯优化进行模型训练和评估，并行化处理。

    参数:
        X (np.ndarray): 特征矩阵。
        y (np.ndarray): 目标变量。
        dataset_name (str): 数据集名称。
        X_test_final (np.ndarray): 最终测试集特征。
        y_test_final (np.ndarray): 最终测试集目标变量。
        models (dict): 模型及其超参数范围。
    """

    # 设置外部交叉验证
    outer = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    log_print(f"外部交叉验证：{outer}", level=logging.INFO)

    # 定义重采样策略和参数搜索空间
    strategys = {
        "SMT": {"C.perc": ['balance', 'extreme'], "k": (5, 15)},
        "RO": {"C.perc": ['balance', 'extreme']},
        "RU": {"C.perc": ['balance', 'extreme']},
        "GN": {"C.perc": ['balance', 'extreme'], "pert": (0.01, 0.3)},
        "SG": {"samp_method": ['balance', 'extreme'], "k": (5, 15), "pert": (0.01, 0.3)},
        "WC": {"C.perc": ['balance', 'extreme'], "over": (0.3, 0.7), "under": (0.3, 0.7)}
    }

    # 准备所有任务
    tasks = []
    for strategy in strategys:
        log_print(f"\n策略: {strategy}", level=logging.INFO)
        param_space_resampling = strategys[strategy]

        for fold, (train_index, test_index) in enumerate(outer.split(X, y), 1):
            tasks.append((strategy, fold, train_index, test_index, X, y, dataset_name))

    log_print(f"总任务数量: {len(tasks)}", level=logging.INFO)
    all_results = []

    # 使用 ProcessPoolExecutor 进行并行处理
    max_workers = min(4, os.cpu_count() or 1)  # 根据CPU核心数调整
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_fold_strategy, task, strategys, models): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                model_results = future.result()
                if model_results is not None:
                    all_results.extend(model_results)
            except Exception as exc:
                strategy, fold, _, _, _, _, _ = task
                log_print(f"{RED}任务失败 - 策略: {strategy}, Fold: {fold}, 错误: {exc}{ENDC}", level=logging.ERROR)

    # 将所有结果转换为 DataFrame
    all_results_df = pd.DataFrame(all_results)

    # 计算每个策略和模型的平均评估指标和平均超参数
    try:
        # 计算评估指标的平均值
        average_results = all_results_df.groupby(['Strategy', 'Model']).agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'R2': 'mean',
            'Pearson_R': 'mean'
        }).reset_index()

        # 计算平均超参数（包括重采样参数和模型参数）
        avg_hyperparameters = all_results_df.groupby(['Strategy', 'Model']).agg({
            'BestParams': lambda x: x.mode()[0],  # 获取最常出现的超参数
            'BestResamplingParams': lambda x: x.mode()[0],
            'BestModelParams': lambda x: x.mode()[0]
        }).reset_index()

        # 保存平均评估结果到 Excel 文件
        average_result_file = os.path.join(base_path, 'average_results.xlsx')
        average_results.to_excel(average_result_file, index=False)
        log_print(f"平均评估结果已保存到 '{average_result_file}'。", level=logging.INFO)

        # 保存平均超参数到 Excel 文件
        avg_hyperparameters_file = os.path.join(base_path, 'average_hyperparameters.xlsx')
        avg_hyperparameters.to_excel(avg_hyperparameters_file, index=False)
        log_print(f"平均超参数已保存到 '{avg_hyperparameters_file}'。", level=logging.INFO)

    except Exception as e:
        log_print(f"保存平均结果或超参数到 Excel 失败，错误: {e}", level=logging.ERROR)

    log_print("所有处理完成。", level=logging.INFO)


def plot_prediction_line(y_true, y_pred):
    """绘制预测值与实际值的对比线图（原始功能）"""
    plt.figure(figsize=(8, 6))
    plt.plot(y_true, y_pred, 'o-', color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel('实际Dmax')
    plt.ylabel('预测Dmax')
    plt.title('预测值与实际值对比线图')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 75)
    plt.ylim(0, 75)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "prediction_line.png"))
    plt.close()


def plot_tuning_curve(scores, params, title, ax):
    """绘制调参结果的曲线图（原始功能）"""
    ax.plot(params, scores, 'o-')
    ax.set_title(title)
    ax.set_xlabel('参数值')
    ax.set_ylabel('分数')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10, weight='bold')


def plot_tuning_curves(mses, maes, r2s, pearsons, params, param_name, model_name):
    """绘制多张性能变化图（原始功能）"""
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name}随{param_name}的变化曲线', weight='bold', size=16)

    plot_tuning_curve(mses, params, f'MSE随{param_name}', ax[0, 0])
    plot_tuning_curve(maes, params, f'MAE随{param_name}', ax[0, 1])
    plot_tuning_curve(r2s, params, f'R²随{param_name}', ax[1, 0])
    plot_tuning_curve(pearsons, params, f'Pearson随{param_name}', ax[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(base_path, f"tuning_curves_{model_name}_{param_name}.png"))
    plt.close()


def plot_model_comparison_bar(final_scores1, final_scores2, final_scores3):
    """绘制柱状对比图（原始功能）"""
    k = 1
    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.1, right=0.98, left=0.06)
    strs = ['MSE', 'MAE', 'R']

    for final_scores in [final_scores1, final_scores2, final_scores3]:
        ax = plt.subplot(1, 3, k)
        x = final_scores.sort_values(strs[k - 1])['Regressors']
        y = final_scores.sort_values(strs[k - 1])[strs[k - 1]]
        ax.bar(x=x, height=y, color='skyblue')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(which='both', direction='in')
        ax.set_xlabel('回归器', fontsize=12, weight='bold')
        ax.set_ylabel(strs[k - 1], fontsize=12, weight='bold')

        for a, b in zip(x, y):
            plt.text(a, b, f"%.4f" % b, ha='center', va='bottom', fontsize=12, weight='bold')

        plt.xticks(weight='bold')
        plt.yticks(weight='bold')
        k += 1
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "model_comparison_bar.png"))
    plt.close()


def plot_learning_curves(pscores, ptrainScore, Sscores, StrainScore, Ascores, AtrainScore):
    """绘制折线图对比图（原始功能）"""
    Scores = [pscores, Sscores, Ascores]
    Tscores = [ptrainScore, StrainScore, AtrainScore]
    k = 1
    strs = ['R', 'mse', 'mae']
    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(wspace=0.4, hspace=0.1, right=0.98, left=0.06)

    for scores, trainscores in zip(Scores, Tscores):
        plt.subplot(1, 3, k)
        plt.plot(list(scores.keys()), list(scores.values()), marker='o', linestyle='-', label='测试集')
        text = [plt.text(i, score + 0.0002, f'{score:.4f}', ha='left', size=10, weight='semibold')
                for i, score in enumerate(scores.values())]
        adjust_text(text)

        plt.plot(list(trainscores.keys()), list(trainscores.values()), marker='o', linestyle='-', label='训练集')
        text = [plt.text(i, score + 0.0002, f'{score:.4f}', ha='left', size=10, weight='semibold')
                for i, score in enumerate(trainscores.values())]
        adjust_text(text)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(which='both', direction='in')

        plt.title('模型学习曲线', size=15, weight='bold')
        plt.ylabel(f'分数 ({strs[k - 1]})', size=12, weight='bold')
        plt.xlabel('模型', size=12, weight='bold')
        plt.tick_params(axis='x', labelsize=10)
        plt.tick_params(axis='y', labelsize=10)
        plt.legend()
        plt.xticks(weight='bold')
        plt.yticks(weight='bold')
        k += 1
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "learning_curves.png"))
    plt.close()


def train_plot(y_true, y_pre, y_scaler, r, model, mse, mae, ax):
    """绘制训练结果散点图（原始功能）"""
    y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pre = y_scaler.inverse_transform(y_pre.reshape(-1, 1))
    sns.scatterplot(x=y_true.reshape(-1), y=y_pre.reshape(-1), palette='Blues', ax=ax)
    sns.regplot(x=y_true.reshape(-1), y=y_pre.reshape(-1), scatter=False, ax=ax)
    ax.plot((0, 75), (0, 75), 'k--')
    ax.set_xlabel('实测Dmax', size=12, weight='bold')
    ax.set_ylabel('预测Dmax', size=12, weight='bold')
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 75)
    ax.text(33.5, 71, f'{model}: R={r:.4f}', ha='center', va='center', size=10, weight='bold')
    ax.text(37.5, 65, f'{model}: MSE={mse:.4f}', ha='center', va='center', size=10, weight='bold')
    ax.text(37.5, 59, f'{model}: MAE={mae:.4f}', ha='center', va='center', size=10, weight='bold')


def plot_multiple_model_results(y_true_list, y_pred_list, y_scaler, model_names, strategy, fold):
    """绘制多个模型的训练结果图（原始功能）"""
    fig, ax = plt.subplots(2, len(model_names), figsize=(18, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for i, (y_true, y_pred, model_name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
        row, col = i // len(model_names), i % len(model_names)
        train_plot(y_true, y_pred, y_scaler, 0.8, model_name, 0.5, 0.3, ax[row, col])  # 示例指标，需替换为真实值

    plt.suptitle(f'{strategy}策略-Fold{fold}多模型预测效果', weight='bold', size=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(base_path, f"multiple_model_results_{strategy}_{fold}.png"))
    plt.close()


def plot_model_performance_bar(results_df):
    """绘制模型性能对比柱状图"""
    if results_df is None or results_df.empty:
        return

    # 按策略和模型分组计算平均R2
    grouped = results_df.groupby(['Strategy', 'Model']).agg({'R2': 'mean'}).reset_index()

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    for i, strategy in enumerate(grouped['Strategy'].unique()):
        subset = grouped[grouped['Strategy'] == strategy]
        plt.subplot(1, len(grouped['Strategy'].unique()), i + 1)
        sns.barplot(x='Model', y='R2', data=subset, palette='viridis')
        plt.title(f"{strategy}策略下模型R2对比", weight='bold')
        plt.ylabel("平均R2分数", weight='bold')
        plt.xticks(weight='bold', rotation=45)

        # 在柱状图上标注数值
        for p in plt.gca().patches:
            height = p.get_height()
            plt.gca().text(p.get_x() + p.get_width() / 2., height + 0.01,
                           f'{height:.4f}', ha="center", weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "model_performance_bar.png"))
    plt.close()


def plot_prediction_scatter(y_true, y_pred, model_name, strategy, fold):
    """绘制预测值与真实值的散点图"""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, palette='Blues')
    sns.regplot(x=y_true, y=y_pred, scatter=False)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)

    # 添加指标注释
    plt.text(y_true.min(), y_true.max() * 0.9,
             f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\nPearson R: {r:.4f}',
             bbox=dict(facecolor='white', alpha=0.8), weight='bold')

    plt.title(f'{strategy}策略-{model_name}-Fold{fold}预测效果', weight='bold')
    plt.xlabel('真实Dmax', weight='bold')
    plt.ylabel('预测Dmax', weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"prediction_scatter_{model_name}_{strategy}_{fold}.png"))
    plt.close()


def plot_ensemble_performance(results_df):
    """绘制模型性能汇总图"""
    if results_df is None or results_df.empty:
        return

    # 按模型分组计算平均指标
    model_metrics = results_df.groupby('Model').agg({
        'R2': 'mean',
        'Pearson_R': 'mean',
        'RMSE': 'mean',
        'MAE': 'mean'
    }).reset_index()

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('模型性能汇总', weight='bold', size=16)

    # R2和Pearson相关系数
    sns.barplot(x='Model', y='R2', data=model_metrics, ax=axes[0, 0], palette='Blues')
    axes[0, 0].set_title('平均R2分数', weight='bold')
    axes[0, 0].set_ylabel('R2', weight='bold')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), weight='bold', rotation=45)

    sns.barplot(x='Model', y='Pearson_R', data=model_metrics, ax=axes[0, 1], palette='Greens')
    axes[0, 1].set_title('平均Pearson相关系数', weight='bold')
    axes[0, 1].set_ylabel('Pearson R', weight='bold')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), weight='bold', rotation=45)

    # RMSE和MAE
    sns.barplot(x='Model', y='RMSE', data=model_metrics, ax=axes[1, 0], palette='Reds')
    axes[1, 0].set_title('平均RMSE', weight='bold')
    axes[1, 0].set_ylabel('RMSE', weight='bold')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), weight='bold', rotation=45)

    sns.barplot(x='Model', y='MAE', data=model_metrics, ax=axes[1, 1], palette='Purples')
    axes[1, 1].set_title('平均MAE', weight='bold')
    axes[1, 1].set_ylabel('MAE', weight='bold')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), weight='bold', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(base_path, "performance_summary.png"))
    plt.close()

def main():
    # 设置日志
    log_file = os.path.join(base_path, 'model_evaluation.log')
    setup_logging(log_file)

    log_print("开始执行主函数...", level=logging.INFO)

    dataset_path = os.path.join(base_path, ".xlsx")
    dataset_name = ""

    log_print(f"数据集路径: {dataset_path}", level=logging.INFO)

    # 读取数据
    try:
        df = pd.read_excel(dataset_path)
        log_print(f"数据读取成功，包含 {df.shape[0]} 条记录和 {df.shape[1]} 个特征。", level=logging.INFO)
    except Exception as e:
        log_print(f"读取数据失败，错误: {e}", level=logging.ERROR)
        return

    # 确保列名正确，假设您的数据集有 'Dmax' 和其他特征列
    actual_columns = list(df.columns)
    log_print(f"数据集列名: {actual_columns}", level=logging.INFO)

    if 'Dmax' not in df.columns:
        log_print(f"数据集中缺少目标变量列 'Dmax'。", level=logging.ERROR)
        return

    # 删除不需要的列，例如 'Alloy' 列
    if 'Alloy' in df.columns:
        df = df.drop(columns=['Alloy'])
        log_print(f"已删除 'Alloy' 列。", level=logging.INFO)

    # 检查是否存在缺失值
    if df.isnull().values.any():
        log_print(f"数据集中存在缺失值，请处理后再运行脚本。", level=logging.ERROR)
        return
    else:
        log_print(f"数据集中不存在缺失值。", level=logging.INFO)

    # 输出各列的数据类型
    log_print(f"数据集各列的数据类型：\n{df.dtypes}", level=logging.INFO)

    # 确认数据类型并转换为 float64
    try:
        df = df.astype(float)
        log_print(f"数据类型已转换为 float。", level=logging.INFO)
    except ValueError as e:
        log_print(f"数据类型转换失败: {e}", level=logging.ERROR)
        return

    # 分离特征和目标变量
    feature_columns = []
    X = df[feature_columns].to_numpy()
    y = df["Dmax"].to_numpy()

    log_print(f"特征矩阵 X 形状: {X.shape}, 目标变量 y 形状: {y.shape}", level=logging.INFO)

    # 数据探索阶段绘图
    plot_data_distribution(y)
    plot_feature_correlation(df, feature_columns)
    plot_correlation_matrix(df, feature_columns)

    # 随机划分训练集和测试集，80%训练集，20%测试集
    try:
        X_train, X_test_final, y_train, y_test_final = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        log_print(f"数据集已随机划分为训练集和测试集。", level=logging.INFO)
        log_print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}", level=logging.INFO)
        log_print(f"测试集形状: X_test={X_test_final.shape}, y_test={y_test_final.shape}", level=logging.INFO)

        # 定义特征列名称列表
        feature_columns = []

    except Exception as e:
        log_print(f"划分训练集和测试集失败，错误: {e}", level=logging.ERROR)
        return

    # 定义自定义模型及其超参数搜索空间
    models = {
        'CautiousWeightedExtraTrees': {
            'model': EAGET,
            'params': {
                'n_estimators': (50, 200),
                'max_depth': (5, 20),
                'min_samples_leaf': (1, 10),
                'min_samples_split': (2, 10),
                'max_features': ['auto', 'sqrt', 'log2'],
                'bootstrap': [True, False],
                'regularization_factor': (0.01, 1.0),
                'learning_rate': (0.01, 0.3),
                'early_stopping_rounds': (5, 20),
                'max_boost_rounds': (10, 100),
                'l1_ratio': (0.0, 1.0),
                'huber_delta': (0.1, 2.0),
                'dropout_rate': (0.0, 0.5),
                'weight_decay': (1e-5, 1e-2)
            }
        }
    }

    # 调用嵌套交叉验证
    repeatedKfold(X_train, y_train, dataset_name, X_test_final, y_test_final, models)


if __name__ == "__main__":
    main()
