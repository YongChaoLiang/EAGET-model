import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyecharts.options as opts
from pyecharts.charts import Scatter, Bar, Pie, Page
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator
from IPython.display import display

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 检查可用GPU
logging.info(f"可用GPU数量: {len(tf.config.list_physical_devices('GPU'))}")


# 自定义稀疏正则化器
class SparsityRegularizer(regularizers.Regularizer):
    def __init__(self, sparsity_param=0.1, beta=1):
        self.sparsity_param = sparsity_param
        self.beta = beta

    def __call__(self, x):
        rho_hat = K.mean(x, axis=0)
        rho = self.sparsity_param
        epsilon = K.epsilon()
        rho_hat = K.clip(rho_hat, epsilon, 1 - epsilon)
        kl_div = rho * K.log(rho / rho_hat) + (1 - rho) * K.log((1 - rho) / (1 - rho_hat))
        return self.beta * K.sum(kl_div)

    def get_config(self):
        return {'sparsity_param': self.sparsity_param, 'beta': self.beta}


def load_and_preprocess_data(data_path):
    """
    加载并预处理数据，包含完整的错误处理和数据清洗流程
    :param data_path: 数据文件路径
    :return: 归一化的稀疏矩阵、DataFrame和成分列名
    """
    if not os.path.exists(data_path):
        logging.error(f"数据文件不存在: {data_path}")
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    try:
        df = pd.read_excel(data_path)
        logging.info(f"成功加载数据，包含 {df.shape[0]} 行 {df.shape[1]} 列")
    except Exception as e:
        logging.error(f"读取数据文件时出错: {e}")
        raise

    # 检查是否存在类别列
    if 'Category' not in df.columns:
        logging.error("数据集中缺少'Category'列，请检查数据文件")
        raise ValueError("数据集中缺少'Category'列，请检查数据文件")

    # 提取成分列
    composition_columns = [col for col in df.columns if col != 'Category']

    # 转换为数值类型并处理缺失值
    df[composition_columns] = df[composition_columns].apply(pd.to_numeric, errors='coerce')
    df[composition_columns] = df[composition_columns].fillna(0)

    # 生成二进制类别标签
    df['Binary_Category'] = df[composition_columns].apply(
        lambda row: generate_binary_category(row, composition_columns), axis=1)

    # 处理稀有类别
    category_counts = df['Binary_Category'].value_counts()
    min_samples = 2
    rare_categories = category_counts[category_counts < min_samples].index
    df['Binary_Category'] = df['Binary_Category'].apply(lambda x: 'Other-Other' if x in rare_categories else x)

    # 准备特征矩阵
    X = df[composition_columns].values

    # 数据有效性检查
    if X.shape[0] == 0 or X.shape[1] == 0:
        logging.error("数据集中没有有效的样本或特征，请检查数据清洗步骤")
        raise ValueError("数据集中没有有效的样本或特征，请检查数据清洗步骤")

    # 归一化处理
    X_sparse = sparse.csr_matrix(X)
    X_normalized = normalize(X_sparse, norm='l2', axis=1)
    X_normalized_sparse = sparse.csr_matrix(X_normalized)

    logging.info(f"数据预处理完成，成分列: {composition_columns}")
    return X_normalized_sparse, df, composition_columns


def generate_binary_category(row, composition_columns):
    """
    基于成分生成二进制类别标签
    :param row: 数据行
    :param composition_columns: 成分列名
    :return: 二进制类别字符串
    """
    elements = {col: row[col] for col in composition_columns if row[col] > 0}

    if not elements:
        return 'Unknown-Unknown'

    # 按含量降序排序
    sorted_elements = sorted(elements.items(), key=lambda item: item[1], reverse=True)
    top_elements = [element[0] for element in sorted_elements[:2]]

    # 确保有两个元素
    if len(top_elements) == 1:
        top_elements.append(top_elements[0])

    return f"{top_elements[0]}-{top_elements[1]}"


def build_sparse_autoencoder(input_dim, hidden_layers=[1024, 512, 256, 128], latent_dim=8,
                             lambda_=1e-5, sparsity_param=0.05, beta=0.5, dropout_rates=[0.2, 0.2, 0.2, 0.2]):
    """
    构建稀疏自编码器模型
    :param input_dim: 输入维度
    :param hidden_layers: 隐藏层单元数列表
    :param latent_dim: 潜空间维度
    :param lambda_: L2正则化系数
    :param sparsity_param: 稀疏参数
    :param beta: 稀疏正则化系数
    :param dropout_rates: 各层Dropout率列表
    :return: 自编码器和编码器模型
    """
    input_layer = layers.Input(shape=(input_dim,), name='encoder_input')
    x = input_layer

    # 编码器部分
    for i, (units, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(lambda_),
                         activity_regularizer=SparsityRegularizer(sparsity_param, beta),
                         name=f'encoder_dense_{i + 1}')(x)
        x = layers.BatchNormalization()(x)  # 添加批归一化
        x = layers.Dropout(dropout_rate, name=f'encoder_dropout_{i + 1}')(x)

    # 潜空间
    latent = layers.Dense(latent_dim, activation='linear', name='latent_space')(x)

    # 解码器部分
    x = latent
    reversed_hidden_layers = hidden_layers[::-1]
    reversed_dropout_rates = dropout_rates[::-1]

    for i, (units, dropout_rate) in enumerate(zip(reversed_hidden_layers, reversed_dropout_rates)):
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(lambda_),
                         name=f'decoder_dense_{i + 1}')(x)
        x = layers.BatchNormalization()(x)  # 添加批归一化
        x = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i + 1}')(x)

    # 输出层
    output_layer = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')(x)

    # 构建模型
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer, name='Sparse_Autoencoder')
    encoder = models.Model(inputs=input_layer, outputs=latent, name='Encoder')

    logging.info(f"自编码器构建完成，输入维度: {input_dim}, 潜空间维度: {latent_dim}")
    return autoencoder, encoder


def visualize_original_data(df, composition_columns, binary_labels):
    """
    原始数据探索性可视化
    :param df: 数据框
    :param composition_columns: 成分列名
    :param binary_labels: 二进制类别标签
    """
    logging.info("开始原始数据可视化...")
    vis_dir = os.path.join("visualizations", "original_data")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 成分分布直方图
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(composition_columns[:6]):  # 只显示前6个成分
        plt.subplot(2, 3, i + 1)
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'{col} 含量分布')
        plt.xlabel(col)
        plt.ylabel('频数')
    plt.tight_layout()
    hist_path = os.path.join(vis_dir, "composition_distributions.png")
    plt.savefig(hist_path, dpi=300)
    logging.info(f"成分分布直方图已保存至: {hist_path}")
    plt.close()

    # 2. 主要成分箱线图
    if len(composition_columns) > 0:
        top_component = composition_columns[0] if len(composition_columns) > 0 else None
        if top_component:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Binary_Category', y=top_component, data=df, palette='Set3')
            plt.title(f'主要成分 {top_component} 在不同类别中的分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            boxplot_path = os.path.join(vis_dir, f"top_component_boxplot.png")
            plt.savefig(boxplot_path, dpi=300)
            logging.info(f"主要成分箱线图已保存至: {boxplot_path}")
            plt.close()

    # 3. 类别分布饼图
    category_counts = pd.Series(binary_labels).value_counts()
    plt.figure(figsize=(10, 8))
    explode = [0.05] * len(category_counts)
    plt.pie(category_counts, labels=category_counts.index, explode=explode,
            autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.axis('equal')  # 保证饼图是圆形
    plt.title('二进制类别分布')
    pie_path = os.path.join(vis_dir, "category_distribution_pie.png")
    plt.savefig(pie_path, dpi=300)
    logging.info(f"类别分布饼图已保存至: {pie_path}")
    plt.close()

    # 4. 成分相关性热图
    if len(composition_columns) <= 20:  # 限制热图大小
        corr = df[composition_columns].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
        plt.title('成分相关性热图')
        corr_path = os.path.join(vis_dir, "composition_correlation_heatmap.png")
        plt.savefig(corr_path, dpi=300)
        logging.info(f"成分相关性热图已保存至: {corr_path}")
        plt.close()

    # 5. 交互式类别分布条形图 (Plotly)
    category_df = pd.DataFrame({'Category': category_counts.index, 'Count': category_counts.values})
    fig = px.bar(category_df, x='Category', y='Count', color='Category',
                 title='二进制类别样本数量',
                 labels={'Count': '样本数量', 'Category': '类别'},
                 color_discrete_sequence=sns.color_palette("viridis", len(category_counts)))
    fig.update_layout(
        xaxis=dict(tickangle=45),
        margin=dict(l=40, r=20, t=50, b=20)
    )
    bar_path = os.path.join(vis_dir, "interactive_category_bar.html")
    fig.write_html(bar_path)
    logging.info(f"交互式类别条形图已保存至: {bar_path}")

    logging.info("原始数据可视化完成")


def visualize_autoencoder_training(history, model, vis_dir):
    """
    自编码器训练过程可视化
    :param history: 训练历史
    :param model: 自编码器模型
    :param vis_dir: 可视化保存目录
    """
    logging.info("开始训练过程可视化...")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title('自编码器训练损失曲线')
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(vis_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    logging.info(f"损失曲线已保存至: {loss_path}")
    plt.close()

    # 2. 学习率变化曲线
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'])
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.title('学习率变化曲线')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        lr_path = os.path.join(vis_dir, "learning_rate_curve.png")
        plt.savefig(lr_path, dpi=300)
        logging.info(f"学习率曲线已保存至: {lr_path}")
        plt.close()

    # 3. 模型结构可视化
    try:
        tf.keras.utils.plot_model(model, to_file=os.path.join(vis_dir, "model_structure.png"),
                                  show_shapes=True, show_layer_names=True)
        logging.info(f"模型结构已保存至: {os.path.join(vis_dir, 'model_structure.png')}")
    except Exception as e:
        logging.warning(f"模型结构可视化失败: {e}")

    # 4. 重建误差分析
    if hasattr(history, 'model') and hasattr(history.model, 'predict'):
        try:
            # 假设X_normalized是归一化后的原始数据
            X_normalized = history.model.inputs[0].numpy()
            y_pred = history.model.predict(X_normalized)
            reconstruction_error = np.mean((X_normalized - y_pred) ** 2, axis=1)

            plt.figure(figsize=(10, 6))
            plt.hist(reconstruction_error, bins=50, color='skyblue', edgecolor='black')
            plt.xlabel('重建误差')
            plt.ylabel('频数')
            plt.title('自编码器重建误差分布')
            plt.grid(True, alpha=0.3)
            recon_error_path = os.path.join(vis_dir, "reconstruction_error_hist.png")
            plt.savefig(recon_error_path, dpi=300)
            logging.info(f"重建误差分布已保存至: {recon_error_path}")
            plt.close()

            # 重建误差与类别的关系
            if hasattr(history, 'df') and 'Binary_Category' in history.df.columns:
                error_df = pd.DataFrame({
                    'Reconstruction_Error': reconstruction_error,
                    'Category': history.df['Binary_Category']
                })

                plt.figure(figsize=(12, 8))
                sns.boxplot(x='Category', y='Reconstruction_Error', data=error_df, palette='Set3')
                plt.title('不同类别重建误差分布')
                plt.xticks(rotation=45)
                plt.tight_layout()
                cat_error_path = os.path.join(vis_dir, "category_reconstruction_error.png")
                plt.savefig(cat_error_path, dpi=300)
                logging.info(f"类别重建误差分布已保存至: {cat_error_path}")
                plt.close()

                # 交互式重建误差分析
                error_df['Sample_ID'] = range(len(error_df))
                fig = px.scatter(error_df, x='Sample_ID', y='Reconstruction_Error',
                                 color='Category', hover_data=['Category'],
                                 title='样本重建误差与类别关系')
                fig.update_layout(
                    xaxis=dict(title='样本ID'),
                    yaxis=dict(title='重建误差'),
                    margin=dict(l=40, r=20, t=50, b=20)
                )
                interactive_error_path = os.path.join(vis_dir, "interactive_reconstruction_error.html")
                fig.write_html(interactive_error_path)
                logging.info(f"交互式重建误差分析已保存至: {interactive_error_path}")

        except Exception as e:
            logging.warning(f"重建误差分析失败: {e}")

    logging.info("训练过程可视化完成")


def visualize_latent_space(encoded_features, df, binary_labels, composition_columns, vis_dir):
    """
    潜空间特征可视化
    :param encoded_features: 编码后的特征
    :param df: 原始数据框
    :param binary_labels: 二进制类别标签
    :param composition_columns: 成分列名
    :param vis_dir: 可视化保存目录
    """
    logging.info("开始潜空间可视化...")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. PCA降维可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(encoded_features)
    df['PCA_1'] = pca_result[:, 0]
    df['PCA_2'] = pca_result[:, 1]

    unique_labels = np.unique(binary_labels)
    num_classes = len(unique_labels)

    # 自定义颜色方案
    if num_classes <= 20:
        custom_palette = sns.color_palette("tab20", num_classes).as_hex()
    else:
        custom_palette = sns.color_palette("hsv", num_classes).as_hex()

    palette_dict = {label: color for label, color in zip(unique_labels, custom_palette)}

    # Matplotlib散点图
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='PCA_1', y='PCA_2',
        hue='Binary_Category',
        palette=palette_dict,
        data=df,
        alpha=0.8,
        s=80
    )
    plt.title('潜空间可视化 (PCA降维)', fontsize=16)
    plt.xlabel('PCA成分 1', fontsize=14)
    plt.ylabel('PCA成分 2', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 调整图例
    if num_classes <= 15:
        plt.legend(title='二进制类别', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.legend(title='二进制类别', bbox_to_anchor=(1.05, 1), loc='upper left',
                   ncol=2, fontsize=8)

    # 添加PCA解释方差
    explained_variance = pca.explained_variance_ratio_
    plt.figtext(0.01, 0.01,
                f'PCA解释方差: 成分1={explained_variance[0]:.4f}, 成分2={explained_variance[1]:.4f}',
                bbox=dict(facecolor='white', alpha=0.8), fontsize=12)

    plt.tight_layout()
    pca_path = os.path.join(vis_dir, "latent_space_pca.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    logging.info(f"PCA潜空间图已保存至: {pca_path}")
    plt.close()

    # 2. t-SNE降维可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=3000)
    tsne_result = tsne.fit_transform(encoded_features)
    df['TSNE_1'] = tsne_result[:, 0]
    df['TSNE_2'] = tsne_result[:, 1]

    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='TSNE_1', y='TSNE_2',
        hue='Binary_Category',
        palette=palette_dict,
        data=df,
        alpha=0.8,
        s=80
    )
    plt.title('潜空间可视化 (t-SNE降维)', fontsize=16)
    plt.xlabel('t-SNE维度 1', fontsize=14)
    plt.ylabel('t-SNE维度 2', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if num_classes <= 15:
        plt.legend(title='二进制类别', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.legend(title='二进制类别', bbox_to_anchor=(1.05, 1), loc='upper left',
                   ncol=2, fontsize=8)

    plt.tight_layout()
    tsne_path = os.path.join(vis_dir, "latent_space_tsne.png")
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    logging.info(f"t-SNE潜空间图已保存至: {tsne_path}")
    plt.close()

    # 3. 3D PCA可视化 (Plotly)
    pca_3d = PCA(n_components=3)
    pca_3d_result = pca_3d.fit_transform(encoded_features)
    df['PCA_3D_1'] = pca_3d_result[:, 0]
    df['PCA_3D_2'] = pca_3d_result[:, 1]
    df['PCA_3D_3'] = pca_3d_result[:, 2]

    fig = px.scatter_3d(df, x='PCA_3D_1', y='PCA_3D_2', z='PCA_3D_3',
                        color='Binary_Category', hover_data=composition_columns[:3],
                        title='3D潜空间可视化 (PCA降维)',
                        color_discrete_sequence=custom_palette)
    fig.update_layout(
        margin=dict(l=40, r=20, t=50, b=20),
        scene=dict(
            xaxis=dict(title='PCA成分 1'),
            yaxis=dict(title='PCA成分 2'),
            zaxis=dict(title='PCA成分 3')
        )
    )
    pca_3d_path = os.path.join(vis_dir, "latent_space_pca_3d.html")
    fig.write_html(pca_3d_path)
    logging.info(f"3D PCA潜空间图已保存至: {pca_3d_path}")

    # 4. 潜空间特征分布
    plt.figure(figsize=(15, 12))
    for i in range(min(8, encoded_features.shape[1])):  # 最多显示8个潜特征
        plt.subplot(2, 4, i + 1)
        sns.histplot(encoded_features[:, i], kde=True, color=custom_palette[i % len(custom_palette)])
        plt.title(f'潜特征 {i + 1} 分布')
        plt.xlabel(f'潜特征 {i + 1} 值')
        plt.ylabel('频数')
    plt.tight_layout()
    latent_dist_path = os.path.join(vis_dir, "latent_features_distribution.png")
    plt.savefig(latent_dist_path, dpi=300)
    logging.info(f"潜特征分布已保存至: {latent_dist_path}")
    plt.close()

    # 5. 潜特征与类别的关系
    latent_df = pd.DataFrame(encoded_features, columns=[f'Latent_{i + 1}' for i in range(encoded_features.shape[1])])
    latent_df['Category'] = binary_labels

    if encoded_features.shape[1] <= 4:
        n_cols = encoded_features.shape[1]
        n_rows = 1
    else:
        n_cols = 4
        n_rows = (encoded_features.shape[1] + 3) // 4  # 向上取整

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    for i in range(encoded_features.shape[1]):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x='Category', y=f'Latent_{i + 1}', data=latent_df, palette='Set3')
        plt.title(f'潜特征 {i + 1} 在不同类别中的分布')
        plt.xticks(rotation=45)
    plt.tight_layout()
    latent_cat_path = os.path.join(vis_dir, "latent_features_by_category.png")
    plt.savefig(latent_cat_path, dpi=300)
    logging.info(f"潜特征类别分布已保存至: {latent_cat_path}")
    plt.close()

    # 6. 潜特征相关性热图
    latent_corr = latent_df.drop('Category', axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(latent_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
    plt.title('潜特征相关性热图')
    corr_path = os.path.join(vis_dir, "latent_features_correlation.png")
    plt.savefig(corr_path, dpi=300)
    logging.info(f"潜特征相关性热图已保存至: {corr_path}")
    plt.close()

    # 7. 交互式潜空间可视化 (pyecharts)
    page = Page()

    # PCA散点图
    scatter_pca = (
        Scatter()
        .add_xaxis(pca_result[:, 0].tolist())
        .add_yaxis(
            series_name="",
            y_axis=pca_result[:, 1].tolist(),
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=[palette_dict.get(label, '#636EFA') for label in binary_labels]
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="PCA潜空间可视化"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(name="PCA成分1"),
            yaxis_opts=opts.AxisOpts(name="PCA成分2"),
            legend_opts=opts.LegendOpts(pos_right="1%")
        )
    )

    # 添加类别标签到图例
    for label in unique_labels:
        scatter_pca.add_yaxis(
            series_name=label,
            y_axis=[],
            x_axis=[],
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=palette_dict.get(label, '#636EFA'))
        )

    page.add(scatter_pca)

    # t-SNE散点图
    scatter_tsne = (
        Scatter()
        .add_xaxis(tsne_result[:, 0].tolist())
        .add_yaxis(
            series_name="",
            y_axis=tsne_result[:, 1].tolist(),
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=[palette_dict.get(label, '#636EFA') for label in binary_labels]
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="t-SNE潜空间可视化"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(name="t-SNE维度1"),
            yaxis_opts=opts.AxisOpts(name="t-SNE维度2"),
            legend_opts=opts.LegendOpts(pos_right="1%")
        )
    )

    for label in unique_labels:
        scatter_tsne.add_yaxis(
            series_name=label,
            y_axis=[],
            x_axis=[],
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=palette_dict.get(label, '#636EFA'))
        )

    page.add(scatter_tsne)

    # 保存交互式可视化页面
    interactive_latent_path = os.path.join(vis_dir, "interactive_latent_space.html")
    page.render(interactive_latent_path)
    logging.info(f"交互式潜空间可视化已保存至: {interactive_latent_path}")

    logging.info("潜空间可视化完成")
    return df, pca_result, tsne_result


def visualize_encoder_activations(encoder, X_sample, composition_columns, vis_dir, num_samples=10):
    """
    可视化编码器各层激活值
    :param encoder: 编码器模型
    :param X_sample: 样本数据
    :param composition_columns: 成分列名
    :param vis_dir: 可视化保存目录
    :param num_samples: 可视化的样本数量
    """
    logging.info("开始编码器激活值可视化...")
    os.makedirs(vis_dir, exist_ok=True)

    if X_sample.shape[0] == 0:
        logging.warning("没有样本数据可用于激活值可视化")
        return

    # 取前num_samples个样本
    X_vis = X_sample[:num_samples]
    if sparse.issparse(X_vis):
        X_vis = X_vis.toarray()

    # 获取各层激活值
    layer_outputs = [layer.output for layer in encoder.layers[1:]]  # 排除输入层
    activation_model = models.Model(inputs=encoder.input, outputs=layer_outputs)
    activations = activation_model.predict(X_vis)

    # 可视化各层激活值
    for i, activation in enumerate(activations):
        layer_name = encoder.layers[i + 1].name  # +1 跳过输入层
        logging.info(f"可视化层: {layer_name}, 形状: {activation.shape}")

        if len(activation.shape) == 2:  # 处理全连接层
            plt.figure(figsize=(15, 10))
            for j in range(min(activation.shape[1], 8)):  # 最多显示8个神经元
                plt.subplot(2, 4, j + 1)
                plt.plot(activation[:, j])
                plt.title(f'层 {layer_name}, 神经元 {j} 激活值')
                plt.xlabel('样本')
                plt.ylabel('激活值')
            plt.tight_layout()
            act_path = os.path.join(vis_dir, f"layer_{layer_name}_activations.png")
            plt.savefig(act_path, dpi=300)
            logging.info(f"层 {layer_name} 激活值已保存至: {act_path}")
            plt.close()

        elif len(activation.shape) == 3:  # 处理卷积层 (虽然这里没有卷积层，但保留此逻辑)
            plt.figure(figsize=(15, 15))
            num_filters = min(activation.shape[3], 16)
            for j in range(num_filters):
                plt.subplot(4, 4, j + 1)
                plt.imshow(activation[0, :, :, j], cmap='viridis')
                plt.title(f'层 {layer_name}, 滤波器 {j}')
                plt.axis('off')
            plt.tight_layout()
            act_path = os.path.join(vis_dir, f"layer_{layer_name}_filters.png")
            plt.savefig(act_path, dpi=300)
            logging.info(f"层 {layer_name} 滤波器已保存至: {act_path}")
            plt.close()

    logging.info("编码器激活值可视化完成")


def visualize_reconstruction(encoder, decoder, X_sample, composition_columns, vis_dir, num_samples=10):
    """
    可视化样本重建效果
    :param encoder: 编码器模型
    :param decoder: 解码器模型
    :param X_sample: 样本数据
    :param composition_columns: 成分列名
    :param vis_dir: 可视化保存目录
    :param num_samples: 可视化的样本数量
    """
    logging.info("开始样本重建可视化...")
    os.makedirs(vis_dir, exist_ok=True)

    if X_sample.shape[0] == 0:
        logging.warning("没有样本数据可用于重建可视化")
        return

    # 取前num_samples个样本
    X_vis = X_sample[:num_samples]
    if sparse.issparse(X_vis):
        X_vis = X_vis.toarray()

    # 重建样本
    X_reconstructed = decoder.predict(encoder.predict(X_vis))

    # 可视化原始样本与重建样本
    n_features = min(len(composition_columns), 10)
    feature_names = composition_columns[:n_features]

    plt.figure(figsize=(15, num_samples * 2))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)

        # 原始样本
        original = X_vis[i, :n_features]
        reconstructed = X_reconstructed[i, :n_features]

        x = np.arange(len(feature_names))
        width = 0.35

        plt.bar(x - width / 2, original, width, label='原始', color='skyblue')
        plt.bar(x + width / 2, reconstructed, width, label='重建', color='lightgreen')

        plt.title(f'样本 {i + 1} 原始与重建成分对比')
        plt.xlabel('成分')
        plt.ylabel('含量')
        plt.xticks(x, feature_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    recon_path = os.path.join(vis_dir, "sample_reconstruction_comparison.png")
    plt.savefig(recon_path, dpi=300)
    logging.info(f"样本重建对比已保存至: {recon_path}")
    plt.close()

    # 重建误差雷达图
    if n_features <= 12:  # 雷达图适合较少的特征
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        # 特征角度
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        for i in range(num_samples):
            original = X_vis[i, :n_features].tolist()
            original += original[:1]  # 闭合图形

            reconstructed = X_reconstructed[i, :n_features].tolist()
            reconstructed += reconstructed[:1]  # 闭合图形

            error = [abs(o - r) for o, r in zip(original, reconstructed)]
            error += error[:1]  # 闭合图形

            ax.plot(angles, error, 'o-', linewidth=2, label=f'样本 {i + 1}')
            ax.fill(angles, error, alpha=0.2)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles), feature_names)
        plt.title('样本重建误差雷达图')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()

        radar_path = os.path.join(vis_dir, "reconstruction_error_radar.png")
        plt.savefig(radar_path, dpi=300)
        logging.info(f"重建误差雷达图已保存至: {radar_path}")
        plt.close()

    logging.info("样本重建可视化完成")


def main():
    """主函数，整合数据处理、模型训练和可视化"""
    logging.info("开始执行主程序...")

    # 创建可视化目录
    vis_dir = os.path.join("visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 数据文件路径
    data_path = input("请输入数据集文件路径: ")

    try:
        # 1. 加载并预处理数据
        X_normalized_sparse, df, composition_columns = load_and_preprocess_data(data_path)
        binary_labels = df['Binary_Category'].values

        # 2. 原始数据可视化
        visualize_original_data(df, composition_columns, binary_labels)

        # 3. 数据集划分
        from collections import Counter
        label_counts = Counter(binary_labels)

        # 尝试分层划分数据集
        try:
            X_train_sparse, X_val_sparse, y_train, y_val = train_test_split(
                X_normalized_sparse, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
            logging.info("成功划分训练集和验证集")
        except ValueError as e:
            logging.error(f"划分训练集和验证集时出错: {e}")
            logging.info("尝试移除稀有类别后重新划分")
            rare_classes = [label for label, count in label_counts.items() if count < 2]
            mask = ~df['Binary_Category'].isin(rare_classes)
            X_filtered_sparse = X_normalized_sparse[mask]
            y_filtered = binary_labels[mask]
            logging.info(f"移除 {len(rare_classes)} 个稀有类别后，剩余样本数: {X_filtered_sparse.shape[0]}")
            new_label_counts = Counter(y_filtered)
            X_train_sparse, X_val_sparse, y_train, y_val = train_test_split(
                X_filtered_sparse, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)
            logging.info("数据集重新划分成功")

        # 转换为密集矩阵用于训练
        X_train = X_train_sparse.toarray()
        X_val = X_val_sparse.toarray()
        input_dim = X_train.shape[1]

        # 4. 构建自编码器
        hidden_layers = [1024, 512, 256, 128]
        latent_dim = 8
        lambda_ = 1e-5
        sparsity_param = 0.05
        beta = 0.5
        dropout_rates = [0.2, 0.2, 0.2, 0.2]

        autoencoder, encoder = build_sparse_autoencoder(
            input_dim, hidden_layers, latent_dim, lambda_, sparsity_param, beta, dropout_rates)

        # 打印模型摘要
        autoencoder.summary()

        # 5. 配置训练回调
        optimizer = optimizers.Adam(learning_rate=1e-4)
        autoencoder.compile(optimizer=optimizer, loss='mse')

        # 回调函数
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=300,
            restore_best_weights=True,
            verbose=1
        )
        checkpoint_dir = os.path.join("checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = ModelCheckpoint(
            os.path.join(checkpoint_dir, "best_autoencoder.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        tensorboard = TensorBoard(
            log_dir=os.path.join("logs", "autoencoder"),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )

        # 6. 训练模型
        history = autoencoder.fit(
            X_train, X_train,
            epochs=1000,
            batch_size=32,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping, lr_scheduler, checkpoint, tensorboard],
            verbose=1
        )

        # 保存训练历史
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(vis_dir, "training_history.csv"), index=False)
        logging.info("训练历史已保存至 training_history.csv")

        # 7. 训练过程可视化
        train_vis_dir = os.path.join(vis_dir, "training")
        visualize_autoencoder_training(history, autoencoder, train_vis_dir)

        # 8. 提取潜空间特征
        encoded_features = encoder.predict(X_normalized_sparse.toarray(), batch_size=32)

        # 9. 潜空间可视化
        latent_vis_dir = os.path.join(vis_dir, "latent_space")
        df, pca_result, tsne_result = visualize_latent_space(
            encoded_features, df, binary_labels, composition_columns, latent_vis_dir)

        # 10. 编码器激活值可视化
        activation_vis_dir = os.path.join(vis_dir, "encoder_activations")
        visualize_encoder_activations(encoder, X_normalized_sparse, composition_columns, activation_vis_dir)

        # 11. 重建效果可视化
        decoder = models.Model(inputs=encoder.input, outputs=autoencoder.output)
        recon_vis_dir = os.path.join(vis_dir, "reconstruction")
        visualize_reconstruction(encoder, decoder, X_normalized_sparse, composition_columns, recon_vis_dir)

        # 12. 保存潜空间特征
        latent_space_df = pd.DataFrame(encoded_features, columns=[f'Latent_{i + 1}' for i in range(latent_dim)])
        latent_space_df['Binary_Category'] = binary_labels
        latent_space_df['PCA_1'] = pca_result[:, 0]
        latent_space_df['PCA_2'] = pca_result[:, 1]
        latent_space_df['TSNE_1'] = tsne_result[:, 0]
        latent_space_df['TSNE_2'] = tsne_result[:, 1]

        tables_dir = os.path.join("tables")
        os.makedirs(tables_dir, exist_ok=True)
        latent_space_path = os.path.join(tables_dir, "latent_space_features.xlsx")
        latent_space_df.to_excel(latent_space_path, index=False)
        logging.info(f"潜空间特征表已保存至: {latent_space_path}")

        logging.info("所有可视化和分析完成")

    except Exception as e:
        logging.error(f"程序执行过程中出错: {e}", exc_info=True)


if __name__ == "__main__":
    main()
