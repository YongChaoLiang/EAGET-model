import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for result reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check available GPUs (optional)
logging.info(f"Number of available GPUs: {len(tf.config.list_physical_devices('GPU'))}")


# Custom sparsity regularizer
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
    Load and preprocess the data.
    :param data_path: Path to the data file.
    :return: Normalized sparse matrix, DataFrame, and composition column names.
    """
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        raise
    if 'Category' not in df.columns:
        logging.error("The dataset is missing the 'Category' column. Please check the data file.")
        raise ValueError("The dataset is missing the 'Category' column. Please check the data file.")
    composition_columns = [col for col in df.columns if col != 'Category']
    labels = df['Category'].values
    df[composition_columns] = df[composition_columns].apply(pd.to_numeric, errors='coerce')
    df[composition_columns] = df[composition_columns].fillna(0)
    df['Binary_Category'] = df[composition_columns].apply(
        lambda row: generate_binary_category(row, composition_columns), axis=1)
    category_counts = df['Binary_Category'].value_counts()
    min_samples = 2
    rare_categories = category_counts[category_counts < min_samples].index
    df['Binary_Category'] = df['Binary_Category'].apply(lambda x: 'Other-Other' if x in rare_categories else x)
    X = df[composition_columns].values
    if X.shape[0] == 0 or X.shape[1] == 0:
        logging.error("There are no valid samples or features in the dataset. Please check the data cleaning steps.")
        raise ValueError("There are no valid samples or features in the dataset. Please check the data cleaning steps.")
    X_sparse = sparse.csr_matrix(X)
    X_normalized = normalize(X_sparse, norm='l2', axis=1)
    X_normalized_sparse = sparse.csr_matrix(X_normalized)
    return X_normalized_sparse, df, composition_columns


def generate_binary_category(row, composition_columns):
    """
    Generate binary category.
    :param row: Data row.
    :param composition_columns: Composition column names.
    :return: Binary category string.
    """
    elements = {col: row[col] for col in composition_columns if row[col] > 0}
    if not elements:
        return 'Unknown-Unknown'
    sorted_elements = sorted(elements.items(), key=lambda item: item[1], reverse=True)
    top_elements = [element[0] for element in sorted_elements[:2]]
    if len(top_elements) == 1:
        top_elements.append(top_elements[0])
    binary_category = f"{top_elements[0]}-{top_elements[1]}"
    return binary_category


def build_sparse_autoencoder(input_dim, hidden_layers=[1024, 512, 256, 128], latent_dim=8,
                             lambda_=1e-5, sparsity_param=0.05, beta=0.5, dropout_rates=[0.2, 0.2, 0.2, 0.2]):
    """
    Build a sparse autoencoder.
    :param input_dim: Input dimension.
    :param hidden_layers: List of hidden layer units.
    :param latent_dim: Latent space dimension.
    :param lambda_: L2 regularization coefficient.
    :param sparsity_param: Sparsity parameter.
    :param beta: Sparsity regularization coefficient.
    :param dropout_rates: List of dropout rates.
    :return: Autoencoder and encoder models.
    """
    input_layer = layers.Input(shape=(input_dim,), name='encoder_input')
    x = input_layer
    for i, (units, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(lambda_),
                         activity_regularizer=SparsityRegularizer(sparsity_param, beta),
                         name=f'encoder_dense_{i + 1}')(x)
        x = layers.Dropout(dropout_rate, name=f'encoder_dropout_{i + 1}')(x)
    latent = layers.Dense(latent_dim, activation='linear', name='latent_space')(x)
    x = latent
    reversed_hidden_layers = hidden_layers[::-1]
    reversed_dropout_rates = dropout_rates[::-1]
    for i, (units, dropout_rate) in enumerate(zip(reversed_hidden_layers, reversed_dropout_rates)):
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(lambda_),
                         name=f'decoder_dense_{i + 1}')(x)
        x = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i + 1}')(x)
    output_layer = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')(x)
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer, name='Sparse_Autoencoder')
    encoder = models.Model(inputs=input_layer, outputs=latent, name='Encoder')
    return autoencoder, encoder


def main():
    """
    Main function.
    """
    data_path = input("Please enter the path to the dataset file: ")
    try:
        X_normalized_sparse, df, composition_columns = load_and_preprocess_data(data_path)
        binary_labels = df['Binary_Category'].values
        from collections import Counter
        label_counts = Counter(binary_labels)
        try:
            X_train_sparse, X_val_sparse, y_train, y_val = train_test_split(
                X_normalized_sparse, binary_labels, test_size=0.2, random_state=42, stratify=binary_labels)
        except ValueError as e:
            logging.error(f"Error occurred when splitting the training and validation sets: {e}")
            logging.info("Trying to remove classes with only one sample and split the data again.")
            rare_classes = [label for label, count in label_counts.items() if count < 2]
            mask = ~df['Binary_Category'].isin(rare_classes)
            X_filtered_sparse = X_normalized_sparse[mask]
            y_filtered = binary_labels[mask]
            logging.info(
                f"After removing {len(rare_classes)} rare classes, the number of remaining samples: {X_filtered_sparse.shape[0]}")
            new_label_counts = Counter(y_filtered)
            X_train_sparse, X_val_sparse, y_train, y_val = train_test_split(
                X_filtered_sparse, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)
            logging.info("Dataset split successfully.")
        X_train = X_train_sparse.toarray()
        X_val = X_val_sparse.toarray()
        input_dim = X_train.shape[1]
        hidden_layers = [1024, 512, 256, 128]
        latent_dim = 8
        lambda_ = 1e-5
        sparsity_param = 0.05
        beta = 0.5
        dropout_rates = [0.2, 0.2, 0.2, 0.2]
        autoencoder, encoder = build_sparse_autoencoder(input_dim, hidden_layers, latent_dim,
                                                        lambda_, sparsity_param, beta, dropout_rates)
        optimizer = optimizers.Adam(learning_rate=1e-4)
        autoencoder.compile(optimizer=optimizer, loss='mse')
        autoencoder.summary()
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=300,
            restore_best_weights=True,
            verbose=1
        )
        history = autoencoder.fit(
            X_train, X_train,
            epochs=1000,
            batch_size=32,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Curves')
        loss_plot_path = os.path.join(plots_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Loss curve has been saved to {loss_plot_path}")
        plt.show()
        encoded_features = encoder.predict(X_normalized_sparse.toarray(), batch_size=32)
        for i in range(latent_dim):
            df[f'Encoded_Feature_{i + 1}'] = encoded_features[:, i]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(encoded_features)
        df['PCA_1'] = pca_result[:, 0]
        df['PCA_2'] = pca_result[:, 1]
        unique_labels = df['Binary_Category'].unique()
        num_classes = len(unique_labels)
        if num_classes <= 20:
            custom_palette = sns.color_palette("tab20", num_classes).as_hex()
        else:
            custom_palette = sns.color_palette("hsv", num_classes).as_hex()
        palette_dict = {label: color for label, color in zip(unique_labels, custom_palette)}
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x='PCA_1', y='PCA_2',
            hue='Binary_Category',
            palette=palette_dict,
            data=df,
            alpha=0.8
        )
        plt.title('Visualization of the Latent Space (PCA Dimensionality Reduction)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Binary Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        pca_plot_path = os.path.join(plots_dir, 'latent_space_pca.png')
        plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"PCA latent space plot has been saved to {pca_plot_path}")
        plt.show()
        latent_space_df = pd.DataFrame({
            'Sample_ID': df.index,
            'PCA_1_basis': pca_result[:, 0],
            'PCA_2_basis': pca_result[:, 1],
            'Binary_Category': df['Binary_Category']
        })
        tables_dir = "Visualization_of_Latent_Space"
        os.makedirs(tables_dir, exist_ok=True)
        latent_space_table_path = os.path.join(tables_dir, 'latent_space_coordinates.xlsx')
        latent_space_df.to_excel(latent_space_table_path, index=False)
        logging.info(f"Latent space coordinate table has been saved to {latent_space_table_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
