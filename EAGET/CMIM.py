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

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_selection.log"),
        logging.StreamHandler()
    ]
)


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def load_and_preprocess_data(file_path, features, target, n_bins=5):
    """
    Load and preprocess data.
    :param file_path: Path to the data file.
    :param features: List of feature column names.
    :param target: Name of the target variable column.
    :param n_bins: Number of bins for discretization (default: 5).
    :return: Preprocessed feature matrix X_scaled and target variable y.
    """
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Successfully loaded data from {file_path}.")
    except FileNotFoundError:
        logging.error(f"File not found at path {file_path}.")
        raise
    except Exception as e:
        logging.error(f"Error occurred while reading the file: {e}")
        raise

    if target not in df.columns:
        logging.error(f"Target column {target} is missing.")
        raise ValueError(f"Target column {target} is missing.")

    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        logging.error(f"Missing feature columns: {missing_features}")
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = df[features].copy()
    y = df[target].copy()

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if numerical_features:
        X.loc[:, numerical_features] = X[numerical_features].fillna(X[numerical_features].median())
        logging.info("Filled missing values in numerical features with the median.")

    if categorical_features:
        X.loc[:, categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])
        logging.info("Filled missing values in categorical features with the mode.")

    if y.isnull().sum() > 0:
        logging.warning("Missing values found in the target variable. These samples will be removed.")
        df = df.dropna(subset=[target])
        X = df[features].copy()
        y = df[target].copy()
        logging.info("Removed samples with missing target variable.")

    scaler = StandardScaler()
    if numerical_features:
        X_scaled_numerical = pd.DataFrame(scaler.fit_transform(X[numerical_features]), columns=numerical_features)
        logging.info("Standardized numerical features.")
    else:
        X_scaled_numerical = pd.DataFrame()
        logging.info("No numerical features to standardize.")

    if categorical_features:
        encoder = OneHotEncoder(drop='first', sparse=False)
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]),
                                 columns=encoder.get_feature_names_out(categorical_features))
        logging.info(f"One-hot encoded categorical features: {categorical_features}")
    else:
        X_encoded = pd.DataFrame()
        logging.info("No categorical features to one-hot encode.")

    if not X_scaled_numerical.empty and not X_encoded.empty:
        X_scaled = pd.concat([X_scaled_numerical, X_encoded], axis=1)
    elif not X_scaled_numerical.empty:
        X_scaled = X_scaled_numerical
    elif not X_encoded.empty:
        X_scaled = X_encoded
    else:
        logging.error("No available features for analysis.")
        raise ValueError("No available features for analysis.")

    return X_scaled, y


def knn_entropy(data, k=5):
    """
    Calculate entropy using k-nearest neighbors method.
    :param data: Input data matrix.
    :param k: Number of neighbors (default: 5).
    :return: Calculated entropy value.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    radii = distances[:, k]
    n = data.shape[0]
    d = data.shape[1]

    volume_unit_ball = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    entropy = -np.mean(np.log(radii)) + np.log(n) + d * np.log(volume_unit_ball)
    return entropy


def compute_mutual_information(X, y, n_splits=3, n_runs=3, n_neighbors=5):
    """
    Compute mutual information between each feature and the target variable using cross-validation.
    :param X: Feature matrix.
    :param y: Target variable.
    :param n_splits: Number of folds for cross-validation (default: 3).
    :param n_runs: Number of cross-validation runs (default: 3).
    :param n_neighbors: Number of neighbors for mutual information calculation (default: 5).
    :return: Series of average mutual information for each feature with the target variable.
    """
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
    return mi_series


def compute_mutual_information_3D(X_i, X_j, Y, k=5):
    """
    Compute the joint mutual information of three variables.
    :param X_i: First feature variable.
    :param X_j: Second feature variable.
    :param Y: Target variable.
    :param k: Number of neighbors (default: 5).
    :return: Calculated joint mutual information value.
    """
    X_i = X_i.values.reshape(-1, 1) if isinstance(X_i, pd.Series) else X_i.reshape(-1, 1)
    X_j = X_j.values.reshape(-1, 1) if isinstance(X_j, pd.Series) else X_j.reshape(-1, 1)
    Y = Y.values.reshape(-1, 1) if isinstance(Y, pd.Series) else Y.reshape(-1, 1)

    X_i_j = np.hstack((X_i, X_j))
    X_i_j_Y = np.hstack((X_i_j, Y))

    H_Xi_Xj = knn_entropy(X_i_j, k)
    H_Y = knn_entropy(Y, k)
    H_Xi_Xj_Y = knn_entropy(X_i_j_Y, k)

    MI = H_Xi_Xj + H_Y - H_Xi_Xj_Y
    return max(0, MI)


def compute_conditional_mutual_info(X, y, feature_i, feature_j, k=5):
    """
    Compute the conditional mutual information I(X_i; Y | X_j).
    :param X: Feature matrix.
    :param y: Target variable.
    :param feature_i: Name of feature i.
    :param feature_j: Name of feature j.
    :param k: Number of neighbors (default: 5).
    :return: Calculated conditional mutual information value.
    """
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


def cmim_selection(X, y, n_select, mi_series):
    """
    Perform feature selection using the CMIM algorithm.
    :param X: Feature matrix.
    :param y: Target variable.
    :param n_select: Number of features to select.
    :param mi_series: Series of mutual information between each feature and the target variable.
    :return: List of selected features.
    """
    mi_sorted = mi_series.sort_values(ascending=False)
    selected_features = [mi_sorted.index[0]]
    candidates = list(set(X.columns) - set(selected_features))

    while len(selected_features) < n_select and candidates:
        min_cmis = []
        for candidate in candidates:
            cmis = [compute_conditional_mutual_info(X, y, candidate, s) for s in selected_features]
            min_cmis.append(min(cmis))
        best_idx = np.argmax([mi_series[c] - min_cmis[i] for i, c in enumerate(candidates)])
        best_candidate = candidates[best_idx]
        selected_features.append(best_candidate)
        candidates.remove(best_candidate)

    return selected_features


def post_cmim_redundancy_check(X, y, selected_features, mi_series,
                               correlation_threshold=0.9,
                               cmi_threshold=0.05,
                               k=5):
    """
    Check and remove redundant features after CMIM feature selection.
    :param X: Feature matrix.
    :param y: Target variable.
    :param selected_features: List of features selected by CMIM.
    :param mi_series: Series of mutual information between each feature and the target variable.
    :param correlation_threshold: Correlation threshold (default: 0.9).
    :param cmi_threshold: Conditional mutual information threshold (default: 0.05).
    :param k: Number of neighbors (default: 5).
    :return: Final list of selected features and set of removed redundant features.
    """
    selected_features_df = X[selected_features]
    correlation_matrix = selected_features_df.corr().abs()

    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            feature_i = correlation_matrix.columns[i]
            feature_j = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            if corr_value > correlation_threshold:
                high_corr_pairs.append((feature_i, feature_j))

    logging.info(f"Highly correlated feature pairs (correlation > {correlation_threshold}):")
    for pair in high_corr_pairs:
        logging.info(pair)

    if high_corr_pairs:
        print("\nFeature pairs with correlation greater than 0.9:")
        for feat1, feat2 in high_corr_pairs:
            corr_value = correlation_matrix.loc[feat1, feat2]
            print(f"- ({feat1}, {feat2}) Correlation: {corr_value:.2f}")

    features_to_remove = set()
    for feature_i, feature_j in high_corr_pairs:
        cmi = compute_conditional_mutual_info(X, y, feature_i, feature_j, k=k)
        logging.info(f"CMI({feature_i}; Y | {feature_j}) = {cmi:.4f}")
        print(f"CMI({feature_i}; Y | {feature_j}) = {cmi:.4f}")

        if cmi < cmi_threshold:
            mi_i = mi_series[feature_i]
            mi_j = mi_series[feature_j]
            if mi_i < mi_j:
                features_to_remove.add(feature_i)
                logging.info(f"Removed redundant feature: {feature_i}")
                print(f"Removed redundant feature: {feature_i}")
            else:
                features_to_remove.add(feature_j)
                logging.info(f"Removed redundant feature: {feature_j}")
                print(f"Removed redundant feature: {feature_j}")
        else:
            logging.info(f"{feature_i} and {feature_j} are complementary, keeping them.")

    final_selected_features = [feat for feat in selected_features if feat not in features_to_remove]

    return final_selected_features, features_to_remove


def main(input_file_path, output_excel_file, selected_features, target,
         correlation_threshold=0.9, cmi_threshold=0.05,
         n_select=10, n_splits=3, n_runs=3, n_neighbors=5):
    """
    Main function to execute the entire feature selection process.
    :param input_file_path: Path to the input data file.
    :param output_excel_file: Path to the output Excel file for results.
    :param selected_features: List of features to consider.
    :param target: Name of the target variable column.
    :param correlation_threshold: Correlation threshold (default: 0.9).
    :param cmi_threshold: Conditional mutual information threshold (default: 0.05).
    :param n_select: Number of features to select (default: 10).
    :param n_splits: Number of folds for cross-validation (default: 3).
    :param n_runs: Number of cross-validation runs (default: 3).
    :param n_neighbors: Number of neighbors for mutual information calculation (default: 5).
    """
    set_random_seed(42)

    try:
        X_scaled, y = load_and_preprocess_data(input_file_path, selected_features, target)
    except FileNotFoundError:
        logging.error(f"Data file {input_file_path} not found.")
        return
    except ValueError as ve:
        logging.error(f"Value error during data preprocessing: {ve}")
        return
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        return

    mi_series = compute_mutual_information(X_scaled, y, n_splits=n_splits, n_runs=n_runs, n_neighbors=n_neighbors)

    logging.info("Mutual information values between features and the target variable:")
    for feature, mi in mi_series.sort_values(ascending=False).items():
        logging.info(f"{feature}: {mi:.4f}")

    print("Mutual information values between features and the target variable:")
    for feature, mi in mi_series.sort_values(ascending=False).items():
        print(f"{feature}: {mi:.4f}")

    cmim_selected_features = cmim_selection(X_scaled, y, n_select, mi_series)

    final_selected_features, features_to_remove = post_cmim_redundancy_check(
        X=X_scaled,
        y=y,
        selected_features=cmim_selected_features,
        mi_series=mi_series,
        correlation_threshold=correlation_threshold,
        cmi_threshold=cmi_threshold
    )

    print("\nFinal selected features (after removing redundancy):")
    for feature in final_selected_features:
        print(f"- {feature}")

    if features_to_remove:
        print("\nRemoved redundant features:")
        for feature in features_to_remove:
            print(f"- {feature}")
    else:
        print("\nNo redundant features were removed.")

    try:
        output_dir = os.path.dirname(output_excel_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with pd.ExcelWriter(output_excel_file) as writer:
            selected_df = pd.DataFrame(final_selected_features, columns=['Selected Features'])
            selected_df.to_excel(writer, sheet_name='Selected Features', index=False)

            if features_to_remove:
                removed_df = pd.DataFrame(list(features_to_remove), columns=['Removed Redundant Features'])
                removed_df.to_excel(writer, sheet_name='Removed Redundant Features', index=False)

            mi_df = mi_series.sort_values(ascending=False).reset_index()
            mi_df.columns = ['Feature', 'Average Mutual Information with Target']
            mi_df.to_excel(writer, sheet_name='MI_Series', index=False)

            correlation_matrix_initial = X_scaled[selected_features].corr().abs()
            correlation_matrix_initial.to_excel(writer, sheet_name='Correlation Matrix')

        logging.info(f"\nFeature selection results have been saved to {output_excel_file}")
    except Exception as e:
        logging.error(f"Error occurred while saving results to Excel: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Feature Selection using CMIM and Redundancy Check')
    parser.add_argument('input_file', type=str, help='Path to the input Excel file containing the dataset.')
    parser.add_argument('output_file', type=str, help='Path to the output Excel file to save the results.')
    parser.add_argument('--features', nargs='+', required=True,
                        help='List of feature names to consider for feature selection.')
    parser.add_argument('--target', type=str, required=True,
                        help='Name of the target variable column in the dataset.')
    parser.add_argument('--corr_threshold', type=float, default=0.9,
                        help='Correlation threshold for identifying highly correlated features (default: 0.9).')
    parser.add_argument('--cmi_threshold', type=float, default=0.05,
                        help='Conditional mutual information threshold for redundancy check (default: 0.05).')
    parser.add_argument('--n_select', type=int, default=10,
                        help='Number of features to select (default: 10).')
    parser.add_argument('--n_splits', type=int, default=3,
                        help='Number of folds for cross-validation (default: 3).')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of cross-validation runs (default: 3).')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='Number of neighbors for mutual information calculation (default: 5).')

    args = parser.parse_args()

    main(
        input_file_path=args.input_file,
        output_excel_file=args.output_file,
        selected_features=args.features,
        target=args.target,
        correlation_threshold=args.corr_threshold,
        cmi_threshold=args.cmi_threshold,
        n_select=args.n_select,
        n_splits=args.n_splits,
        n_runs=args.n_runs,
        n_neighbors=args.n_neighbors
    )