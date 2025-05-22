import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import math
import logging
from sklearn.model_selection import KFold
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_selection.log"),
        logging.StreamHandler()
    ]
)


def set_random_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)


def load_and_preprocess_data(file_path, features, target, n_bins=5):
    """
    加载并预处理数据，包含数据清洗逻辑
    - 移除缺失关键参数的样本
    - 处理重复合金成分（保留最大Dmax）
    - 温度参数取平均值
    """
    try:
        df = pd.read_excel(file_path)
        logging.info(f"成功从 {file_path} 加载数据，包含 {df.shape[0]} 行 {df.shape[1]} 列")

        # 数据分布可视化
        visualize_original_data(df, features, target)

    except FileNotFoundError:
        logging.error(f"在路径 {file_path} 未找到文件")
        raise
    except Exception as e:
        logging.error(f"读取文件时发生错误: {e}")
        raise

    if target not in df.columns:
        logging.error(f"目标列 {target} 缺失")
        raise ValueError(f"目标列 {target} 缺失")

    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        logging.error(f"缺失特征列: {missing_features}")
        raise ValueError(f"缺失特征列: {missing_features}")

    X = df[features].copy()
    y = df[target].copy()

    # 文档中描述的重复样本处理：保留最大Dmax值（假设target为Dmax）
    if target == 'Dmax':
        X = X.groupby(features).max().reset_index()  # 假设特征组合唯一标识合金
        y = X[target]
        X = X.drop(columns=[target])
        logging.info(f"基于{target}去重后，剩余 {X.shape[0]} 个唯一合金样本")

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if numerical_features:
        X.loc[:, numerical_features] = X[numerical_features].fillna(X[numerical_features].median())
        logging.info("用中位数填充数值特征的缺失值")

        # 数值特征分布可视化
        visualize_numerical_distribution(X, numerical_features)

    if categorical_features:
        X.loc[:, categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])
        logging.info("用众数填充分类特征的缺失值")

        # 分类特征分布可视化
        visualize_categorical_distribution(X, categorical_features)

    if y.isnull().sum() > 0:
        logging.warning("目标变量中发现缺失值，将移除这些样本")
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]
        logging.info(f"移除缺失值后，剩余 {X.shape[0]} 个样本")

    scaler = StandardScaler()
    if numerical_features:
        X_scaled_numerical = pd.DataFrame(scaler.fit_transform(X[numerical_features]), columns=numerical_features)
        logging.info("标准化数值特征")
    else:
        X_scaled_numerical = pd.DataFrame()
        logging.info("没有需要标准化的数值特征")

    if categorical_features:
        encoder = OneHotEncoder(drop='first', sparse=False)
        X_encoded = pd.DataFrame(
            encoder.fit_transform(X[categorical_features]),
            columns=encoder.get_feature_names_out(categorical_features)
        )
        logging.info(f"对分类特征进行独热编码: {categorical_features}")
    else:
        X_encoded = pd.DataFrame()
        logging.info("没有需要独热编码的分类特征")

    if not X_scaled_numerical.empty and not X_encoded.empty:
        X_scaled = pd.concat([X_scaled_numerical, X_encoded], axis=1)
    elif not X_scaled_numerical.empty:
        X_scaled = X_scaled_numerical
    elif not X_encoded.empty:
        X_scaled = X_encoded
    else:
        logging.error("没有可用的特征进行分析")
        raise ValueError("没有可用的特征进行分析")

    # 特征相关性热图
    visualize_correlation_matrix(X_scaled, features)

    return X_scaled, y


def visualize_original_data(df, features, target):
    """可视化原始数据的基本分布"""
    vis_dir = os.path.join(os.path.dirname(df), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 目标变量分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target], kde=True, color='skyblue')
    plt.title(f"目标变量 {target} 分布直方图")
    plt.xlabel(target)
    plt.ylabel("频数")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"target_distribution_{target}.png"))
    plt.close()

    # 特征与目标变量的散点图（前4个数值特征）
    numerical_features = df[features].select_dtypes(include=[np.number]).columns.tolist()[:4]
    if numerical_features:
        plt.figure(figsize=(12, 8))
        for i, feat in enumerate(numerical_features):
            plt.subplot(2, 2, i + 1)
            sns.scatterplot(x=feat, y=target, data=df)
            sns.regplot(x=feat, y=target, data=df, scatter=False, color='red')
            plt.title(f"{feat} 与 {target} 的关系")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "feature_target_scatter.png"))
        plt.close()


def visualize_numerical_distribution(X, numerical_features):
    """可视化数值特征分布"""
    vis_dir = os.path.join(os.path.dirname(X), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(numerical_features[:6]):  # 最多显示6个特征
        plt.subplot(2, 3, i + 1)
        sns.histplot(X[feat], kde=True, color='skyblue')
        plt.title(f"{feat} 数值分布")
        plt.xlabel(feat)
        plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "numerical_features_distribution.png"))
    plt.close()


def visualize_categorical_distribution(X, categorical_features):
    """可视化分类特征分布"""
    vis_dir = os.path.join(os.path.dirname(X), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(categorical_features[:6]):  # 最多显示6个特征
        plt.subplot(2, 3, i + 1)
        X[feat].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f"{feat} 类别分布")
        plt.xlabel(feat)
        plt.ylabel("频数")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "categorical_features_distribution.png"))
    plt.close()


def visualize_correlation_matrix(X, features):
    """可视化特征相关性矩阵"""
    vis_dir = os.path.join(os.path.dirname(X), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 只处理数值特征
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numerical_features:
        return

    corr = X[numerical_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
    plt.title("特征相关性热图")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "feature_correlation_heatmap.png"))
    plt.close()


def knn_entropy(data, k=5):
    """使用k近邻法计算熵"""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    radii = distances[:, k]
    n = data.shape[0]
    d = data.shape[1]

    volume_unit_ball = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    entropy = -np.mean(np.log(radii)) + np.log(n) + d * np.log(volume_unit_ball)
    return entropy


def compute_mutual_information(X, y, n_splits=3, n_runs=3, n_neighbors=5):
    """计算特征与目标的互信息，包含交叉验证"""
    mi_dict = defaultdict(list)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for run in range(n_runs):
        for train_index, test_index in kf.split(X):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            mi = mutual_info_regression(X_train, y_train, n_neighbors=n_neighbors, random_state=42)
            mi_series = pd.Series(mi, index=X.columns)
            for feature, mi_value in mi_series.items():
                mi_dict[feature].append(mi_value)

    mi_series = {feature: np.mean(values) for feature, values in mi_dict.items()}
    mi_series = pd.Series(mi_series).sort_values(ascending=False)

    # 互信息排序可视化
    visualize_mutual_information(mi_series)

    return mi_series


def visualize_mutual_information(mi_series):
    """可视化互信息排序"""
    vis_dir = os.path.join(os.path.dirname(mi_series.name), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    top_features = mi_series.head(15)  # 显示前15个特征
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_features.values, y=top_features.index, color='skyblue')
    plt.title("特征与目标变量的互信息排序")
    plt.xlabel("互信息值")
    plt.ylabel("特征名称")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "mutual_information_ranking.png"))
    plt.close()


def compute_conditional_mutual_info(X, y, feature_i, feature_j, k=5):
    """计算条件互信息 I(X_i; Y | X_j)"""
    X_i = X[feature_i].values.reshape(-1, 1)
    X_j = X[feature_j].values.reshape(-1, 1)
    Y = y.values.reshape(-1, 1)

    X_i_j = np.hstack((X_i, X_j))
    X_j_Y = np.hstack((X_j, Y))
    X_i_j_Y = np.hstack((X_i_j, Y))

    H_Xi_Xj = knn_entropy(X_i_j, k)
    H_Xj_Y = knn_entropy(X_j_Y, k)
    H_Xj = knn_entropy(X_j, k)
    H_Xi_Xj_Y = knn_entropy(X_i_j_Y, k)

    CMI = H_Xi_Xj + H_Xj_Y - H_Xj - H_Xi_Xj_Y
    return max(0, CMI)


def cami_selection(X, y, n_select, mi_series, lambda_=0.5):
    """实现CAMI特征选择算法"""
    mi_sorted = mi_series.sort_values(ascending=False)
    selected_features = [mi_sorted.index[0]]
    candidates = list(set(X.columns) - set(selected_features))
    logging.info(f"初始选择特征: {selected_features[0]}，MI值: {mi_sorted[0]:.4f}")

    # 记录选择过程
    selection_history = {
        'step': [],
        'feature': [],
        'mi': [],
        'cami_score': []
    }

    while len(selected_features) < n_select and candidates:
        cami_scores = []
        for candidate in candidates:
            # 计算CMI总和
            cmis = [compute_conditional_mutual_info(X, y, candidate, s) for s in selected_features]
            # CAMI核心公式：MI - λ * sum(CMI)
            cami_score = mi_series[candidate] - lambda_ * np.sum(cmis)
            cami_scores.append(cami_score)

        best_idx = np.argmax(cami_scores)
        best_candidate = candidates[best_idx]
        selected_features.append(best_candidate)
        candidates.remove(best_candidate)
        logging.info(f"新增选择特征: {best_candidate}，CAMI分数: {cami_scores[best_idx]:.4f}")

        # 记录选择历史
        selection_history['step'].append(len(selected_features))
        selection_history['feature'].append(best_candidate)
        selection_history['mi'].append(mi_series[best_candidate])
        selection_history['cami_score'].append(cami_scores[best_idx])

    # 可视化选择过程
    visualize_selection_process(pd.DataFrame(selection_history))

    return selected_features


def visualize_selection_process(history_df):
    """可视化CAMI选择过程"""
    if history_df.empty:
        return

    vis_dir = os.path.join(os.path.dirname(history_df), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 特征MI值变化
    ax1.plot(history_df['step'], history_df['mi'], 'o-', color='skyblue')
    ax1.set_ylabel("特征MI值")
    ax1.set_title("CAMI特征选择过程")
    ax1.grid(True, alpha=0.3)

    # CAMI分数变化
    ax2.plot(history_df['step'], history_df['cami_score'], 'o-', color='red')
    ax2.set_xlabel("选择步骤")
    ax2.set_ylabel("CAMI分数")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "cami_selection_process.png"))
    plt.close()


def post_cami_redundancy_check(X, y, selected_features, mi_series, cmi_threshold=0.05, k=5):
    """CAMI冗余检查"""
    features_to_remove = set()
    cmi_matrix = np.zeros((len(selected_features), len(selected_features)))

    # 计算所有特征对的CMI并可视化
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            feature_i = selected_features[i]
            feature_j = selected_features[j]
            cmi = compute_conditional_mutual_info(X, y, feature_i, feature_j, k=k)
            cmi_matrix[i, j] = cmi
            cmi_matrix[j, i] = cmi

            if cmi < cmi_threshold:
                mi_i = mi_series[feature_i]
                mi_j = mi_series[feature_j]
                if mi_i < mi_j:
                    features_to_remove.add(feature_i)
                    logging.info(f"移除冗余特征: {feature_i}，CMI: {cmi:.4f} < {cmi_threshold}")
                else:
                    features_to_remove.add(feature_j)
                    logging.info(f"移除冗余特征: {feature_j}，CMI: {cmi:.4f} < {cmi_threshold}")
            else:
                logging.info(f"保留互补特征对: {feature_i} 和 {feature_j}，CMI: {cmi:.4f} ≥ {cmi_threshold}")

    # 可视化CMI矩阵
    visualize_cmi_matrix(selected_features, cmi_matrix, cmi_threshold)

    final_selected_features = [feat for feat in selected_features if feat not in features_to_remove]
    return final_selected_features, features_to_remove


def visualize_cmi_matrix(features, cmi_matrix, threshold):
    """可视化条件互信息矩阵"""
    vis_dir = os.path.join(os.path.dirname(features[0]), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cmi_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                mask=(cmi_matrix >= threshold), cbar_kws={"label": "条件互信息值"})
    plt.title("特征对条件互信息矩阵")
    plt.xticks(range(len(features)), features, rotation=45)
    plt.yticks(range(len(features)), features)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "cmi_matrix_heatmap.png"))
    plt.close()


def main(input_file_path, output_excel_file, selected_features, target,
         lambda_=0.5, cmi_threshold=0.05,
         n_select=10, n_splits=3, n_runs=3, n_neighbors=5):
    """主函数：执行完整的CAMI特征选择流程"""
    set_random_seed(42)
    vis_dir = os.path.join(os.path.dirname(output_excel_file), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    try:
        X_scaled, y = load_and_preprocess_data(input_file_path, selected_features, target)
    except FileNotFoundError:
        logging.error(f"数据文件 {input_file_path} 未找到")
        return
    except ValueError as ve:
        logging.error(f"数据预处理时发生值错误: {ve}")
        return
    except Exception as e:
        logging.error(f"数据预处理失败: {e}")
        return

    # 计算互信息
    mi_series = compute_mutual_information(X_scaled, y, n_splits=n_splits, n_runs=n_runs, n_neighbors=n_neighbors)
    logging.info("特征与目标变量的互信息值:")
    for feature, mi in mi_series.head(10).items():
        logging.info(f"{feature}: {mi:.4f}")

    print("特征与目标变量的互信息值:")
    for feature, mi in mi_series.head(10).items():
        print(f"{feature}: {mi:.4f}")

    # 执行CAMI特征选择
    cami_selected_features = cami_selection(X_scaled, y, n_select, mi_series, lambda_=lambda_)
    logging.info(f"CAMI初始选择的{len(cami_selected_features)}个特征: {cami_selected_features}")

    # 执行CAMI冗余检查
    final_selected_features, features_to_remove = post_cami_redundancy_check(
        X=X_scaled,
        y=y,
        selected_features=cami_selected_features,
        mi_series=mi_series,
        cmi_threshold=cmi_threshold
    )

    print(f"\n最终通过CAMI选择的{len(final_selected_features)}个特征:")
    for feature in final_selected_features:
        print(f"- {feature}")

    if features_to_remove:
        print(f"\n移除的{len(features_to_remove)}个冗余特征:")
        for feature in features_to_remove:
            print(f"- {feature}")
    else:
        print("\n未发现冗余特征")

    # 可视化最终选择结果
    visualize_final_selection(X_scaled, final_selected_features, mi_series)

    # 保存结果到Excel
    try:
        output_dir = os.path.dirname(output_excel_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with pd.ExcelWriter(output_excel_file) as writer:
            # 保存最终选择的特征
            selected_df = pd.DataFrame(final_selected_features, columns=['Selected Features'])
            selected_df.to_excel(writer, sheet_name='CAMI_Selected_Features', index=False)

            # 保存移除的冗余特征
            if features_to_remove:
                removed_df = pd.DataFrame(list(features_to_remove), columns=['Removed_Redundant_Features'])
                removed_df.to_excel(writer, sheet_name='Removed_Features', index=False)

            # 保存互信息排序
            mi_df = mi_series.sort_values(ascending=False).reset_index()
            mi_df.columns = ['Feature', 'Average_Mutual_Information']
            mi_df.to_excel(writer, sheet_name='MI_Series', index=False)

            # 保存最终特征的相关性矩阵
            if not final_selected_features:
                logging.warning("没有选择任何特征，无法生成相关性矩阵")
            else:
                correlation_matrix = X_scaled[final_selected_features].corr().abs()
                correlation_matrix.to_excel(writer, sheet_name='Correlation_Matrix', index=True)

        logging.info(f"\nCAMI特征选择结果已保存至 {output_excel_file}")
    except Exception as e:
        logging.error(f"保存结果到Excel时发生错误: {e}")


def visualize_final_selection(X, selected_features, mi_series):
    """可视化最终选择的特征"""
    if not selected_features:
        return

    vis_dir = os.path.join(os.path.dirname(X), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 最终特征的互信息条形图
    final_mi = mi_series[selected_features]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=final_mi.values, y=final_mi.index, color='skyblue')
    plt.title("最终选择特征的互信息值")
    plt.xlabel("互信息值")
    plt.ylabel("特征名称")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "final_selected_features_mi.png"))
    plt.close()

    # 特征重要性雷达图
    if len(selected_features) <= 10:  # 雷达图适合特征数较少的情况
        angles = np.linspace(0, 2 * np.pi, len(selected_features), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        values = final_mi.values.tolist()
        values += values[:1]  # 闭合图形

        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), selected_features)
        plt.title("最终选择特征的互信息雷达图")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "final_features_radar.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于CAMI的特征选择算法实现（含可视化）')
    parser.add_argument('input_file', type=str, help='输入Excel数据集路径')
    parser.add_argument('output_file', type=str, help='输出结果Excel路径')
    parser.add_argument('--features', nargs='+', required=True, help='待选择的特征列名列表')
    parser.add_argument('--target', type=str, required=True, help='目标变量列名')
    parser.add_argument('--lambda', type=float, default=0.5, help='CAMI算法的平衡参数λ (默认: 0.5)')
    parser.add_argument('--cmi_threshold', type=float, default=0.05, help='条件互信息冗余阈值 (默认: 0.05)')
    parser.add_argument('--n_select', type=int, default=10, help='选择的特征数量 (默认: 10)')
    parser.add_argument('--n_splits', type=int, default=3, help='交叉验证折数 (默认: 3)')
    parser.add_argument('--n_runs', type=int, default=3, help='交叉验证运行次数 (默认: 3)')
    parser.add_argument('--n_neighbors', type=int, default=5, help='互信息计算的邻居数 (默认: 5)')

    args = parser.parse_args()

    main(
        input_file_path=args.input_file,
        output_excel_file=args.output_file,
        selected_features=args.features,
        target=args.target,
        lambda_=args.
    lambda ,
           cmi_threshold=args.cmi_threshold,
           n_select=args.n_select,
           n_splits=args.n_splits,
           n_runs=args.n_runs,
           n_neighbors=args.n_neighbors
    )
