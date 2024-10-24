import numpy as np
import pandas as pd
import re
import random
import os, logging
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, root_mean_squared_error, mean_absolute_error, r2_score,
    f1_score, precision_score, recall_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numexpr as ne
from sklearn.neighbors import NearestNeighbors

import warnings
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning  # Import UndefinedMetricWarning
# Suppress specific warnings from sklearn and xgboost
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


def load_and_preprocess_data(
    filepath,
    target_columns,
    target_types,
    column_mapping=None,
    categorical_cols=None,
    missing_value_strategy='drop',
    imputation_values=None,
    rows_to_remove=None,
    columns_to_remove=None,
    regex_columns_to_remove=None
):
    """
    Loads data from a CSV file, renames columns, preprocesses the data, and splits into features and targets.

    Parameters:
    - filepath (str): Path to the CSV file.
    - target_columns (list): List of target column names.
    - target_types (dict): Dictionary mapping target column names to their types ('classification' or 'regression').
    - column_mapping (dict, optional): Dictionary for renaming columns.
    - categorical_cols (list, optional): List of categorical column names.
    - missing_value_strategy (str, optional): Strategy to handle missing values. Options: 'drop', 'impute'.
    - imputation_values (dict, optional): Dictionary specifying the imputation values for columns.
    - rows_to_remove (list, optional): List of indices of rows to remove.
    - columns_to_remove (list, optional): List of columns to remove.
    - regex_columns_to_remove (list, optional): Regular expressions to match columns to remove.

    Returns:
    - X (pd.DataFrame): Features.
    - y_dict (dict): Dictionary of target variables.
    """
    df = pd.read_csv(filepath)
    logger.info("Initial Data:")
    logger.info(df.head())

    # Rename columns
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info("\nData after renaming columns:")
        logger.info(df.head())

    # Handle missing values
    if missing_value_strategy == 'drop':
        df = df.dropna()
    elif missing_value_strategy == 'impute':
        # Impute missing values
        if imputation_values is not None:
            df = df.fillna(value=imputation_values)
        else:
            # If no imputation values provided, fill numerical columns with zero, categorical with -1
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('0')
                else:
                    df[col] = df[col].fillna(0)
    else:
        raise ValueError("Invalid missing_value_strategy. Options are 'drop' or 'impute'.")
    logger.info("\nData after handling missing values:")
    logger.info(df.head())

    # Remove duplicates
    df = df.drop_duplicates()
    logger.info("\nData after removing duplicates:")
    logger.info(df.head())

    # Remove specific rows
    if rows_to_remove is not None:
        rows_removed = df.loc[rows_to_remove]
        df = df.drop(index=rows_to_remove)
        logger.info("\nRemoved rows:")
        logger.info(rows_removed)

    # Remove specific columns by exact names
    if columns_to_remove is not None:
        # Verify columns exist before dropping
        existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        if existing_columns_to_remove:
            df = df.drop(columns=existing_columns_to_remove)
            logger.info("\nData after removing specific columns:")
            logger.info(df.head())
        else:
            logger.info("\nNo exact columns matched for removal.")

    # Remove specific columns by regex
    if regex_columns_to_remove is not None:
        cols_before = set(df.columns)
        for regex in regex_columns_to_remove:
            matched_cols = df.filter(regex=regex).columns.tolist()
            if matched_cols:
                df = df.drop(columns=matched_cols)
                logger.info(f"\nRemoved columns matching regex '{regex}': {matched_cols}")
        cols_after = set(df.columns)
        if cols_before != cols_after:
            logger.info("\nData after removing columns with regex:")
            logger.info(df.head())
        else:
            logger.info("\nNo regex columns matched for removal.")


    # Check for columns with constant values
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    if constant_columns:
        logger.info("\nColumns with constant values:")
        logger.info(constant_columns)
        # Drop columns with only constant values
        df = df.drop(columns=constant_columns)
        logger.info("\nData after dropping columns with constant values:")
        logger.info(df.head())

    # Split features and targets
    y_dict = {target: df[target] for target in target_columns}
    X = df.drop(columns=target_columns)
    logger.info("\nFinal Features (X):")
    logger.info(X.head())
    logger.info("\nFinal Targets (y_dict):")
    for target, values in y_dict.items():
        logger.info(f"{target}:")
        logger.info(values.head())

    return X, y_dict


def get_unique_kfold_splits(X, n_splits=5, n_repeats=3, y_cat=None):
    """
    Generates unique KFold splits across multiple repeats.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - n_splits (int): Number of splits.
    - n_repeats (int): Number of repeats.
    - y_cat (pd.Series, optional): Target variable for StratifiedKFold if classification.

    Returns:
    - splits (list): List of (train_index, test_index) tuples.
    """
    splits = []
    used_test_indices = []

    if y_cat is None:
        repeat = 0
        while len(splits) < n_splits * n_repeats:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
            for train_index, test_index in kf.split(X):
                # Ensure uniqueness of test indices sets across splits
                sorted_test_indices = tuple(sorted(test_index))
                if sorted_test_indices not in used_test_indices:
                    splits.append((train_index, test_index))
                    used_test_indices.append(sorted_test_indices)
                    if len(splits) >= n_splits * n_repeats:
                        break
            repeat += 1
            if repeat > 50:
                raise ValueError("Could not generate unique splits. Try reducing n_splits or n_repeats or include more data.")
    else:
        try:
            if not isinstance(y_cat, pd.Series):
                y_cat = pd.Series(y_cat)
        except:
            raise ValueError("y_cat must be a pandas Series")
        repeat = 0
        while len(splits) < n_splits * n_repeats:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
            for train_index, test_index in kf.split(X, y_cat):
                # Ensure uniqueness of test indices sets across splits
                sorted_test_indices = tuple(sorted(test_index))
                if sorted_test_indices not in used_test_indices:
                    splits.append((train_index, test_index))
                    used_test_indices.append(sorted_test_indices)
                    if len(splits) >= n_splits * n_repeats:
                        break
            repeat += 1
            if repeat > 50:
                raise ValueError("Could not generate unique splits. Try reducing n_splits or n_repeats or include more data.")
    return splits


def train_models_and_collect_predictions(X, y, model_type='classification', splits=None):
    """
    Trains models and collects predictions.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - y (pd.Series): Target variable.
    - model_type (str): 'classification' or 'regression'.
    - splits (list): List of (train_index, test_index) splits.

    Returns:
    - predictions_list_train (list): List of DataFrames with train predictions per fold.
    - predictions_list_test (list): List of DataFrames with test predictions per fold.
    - models_dict (dict): Dictionary mapping model types to their trained models.
    """
    if splits is None:
        # Default to KFold with 5 splits
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X))

    models_dict = {'LR': [], 'RF': [], 'XGB': []} 
    predictions_list_test = []
    predictions_list_train = []

    for fold_num, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_type == 'classification':
            # Models: Logistic Regression, Random Forest Classifier, XGBoost Classifier
            model1 = LogisticRegression(max_iter=1000, random_state=42)
            model2 = RandomForestClassifier(random_state=42)
            model3 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

            model1.fit(X_train, y_train)
            model2.fit(X_train, y_train)
            model3.fit(X_train, y_train)

            # Store models
            models_dict['LR'].append(model1)
            models_dict['RF'].append(model2)
            models_dict['XGB'].append(model3)

            # Collect predictions
            prob1_test = model1.predict_proba(X_test)
            prob2_test = model2.predict_proba(X_test)
            prob3_test = model3.predict_proba(X_test)

            prob1_train = model1.predict_proba(X_train)
            prob2_train = model2.predict_proba(X_train)
            prob3_train = model3.predict_proba(X_train)

            # For each model, create a DataFrame of predictions for test set
            df_pred1_test = pd.DataFrame(prob1_test, columns=[f'lr_fold_{fold_num+1}_class_{i}' for i in range(prob1_test.shape[1])], index=X_test.index)
            df_pred2_test = pd.DataFrame(prob2_test, columns=[f'rf_fold_{fold_num+1}_class_{i}' for i in range(prob2_test.shape[1])], index=X_test.index)
            df_pred3_test = pd.DataFrame(prob3_test, columns=[f'xgb_fold_{fold_num+1}_class_{i}' for i in range(prob3_test.shape[1])], index=X_test.index)

            # For each model, create a DataFrame of predictions for train set
            df_pred1_train = pd.DataFrame(prob1_train, columns=[f'lr_fold_{fold_num+1}_class_{i}' for i in range(prob1_train.shape[1])], index=X_train.index)
            df_pred2_train = pd.DataFrame(prob2_train, columns=[f'rf_fold_{fold_num+1}_class_{i}' for i in range(prob2_train.shape[1])], index=X_train.index)
            df_pred3_train = pd.DataFrame(prob3_train, columns=[f'xgb_fold_{fold_num+1}_class_{i}' for i in range(prob3_train.shape[1])], index=X_train.index)

            # Collect true labels
            df_true_test = pd.DataFrame({'true_label': y_test}, index=X_test.index)
            df_true_train = pd.DataFrame({'true_label': y_train}, index=X_train.index)

            # Concatenate predictions and true labels for test set
            df_fold_test = pd.concat([df_true_test, df_pred1_test, df_pred2_test, df_pred3_test], axis=1)

            # Concatenate predictions and true labels for train set
            df_fold_train = pd.concat([df_true_train, df_pred1_train, df_pred2_train, df_pred3_train], axis=1)

            predictions_list_test.append(df_fold_test)
            predictions_list_train.append(df_fold_train)

        elif model_type == 'regression':
            # Models: Linear Regression, Random Forest Regressor, XGBoost Regressor
            model1 = LinearRegression()
            model2 = RandomForestRegressor(random_state=42)
            model3 = xgb.XGBRegressor(random_state=42)

            model1.fit(X_train, y_train)
            model2.fit(X_train, y_train)
            model3.fit(X_train, y_train)

            # Store models
            models_dict['LR'].append(model1)
            models_dict['RF'].append(model2)
            models_dict['XGB'].append(model3)

            # Collect predictions for test set
            pred1_test = model1.predict(X_test)
            pred2_test = model2.predict(X_test)
            pred3_test = model3.predict(X_test)

            # Collect predictions for train set
            pred1_train = model1.predict(X_train)
            pred2_train = model2.predict(X_train)
            pred3_train = model3.predict(X_train)

            # For each model, create a DataFrame of predictions for test set
            df_pred1_test = pd.DataFrame({f'lr_fold_{fold_num+1}': pred1_test}, index=X_test.index)
            df_pred2_test = pd.DataFrame({f'rf_fold_{fold_num+1}': pred2_test}, index=X_test.index)
            df_pred3_test = pd.DataFrame({f'xgb_fold_{fold_num+1}': pred3_test}, index=X_test.index)

            # For each model, create a DataFrame of predictions for train set
            df_pred1_train = pd.DataFrame({f'lr_fold_{fold_num+1}': pred1_train}, index=X_train.index)
            df_pred2_train = pd.DataFrame({f'rf_fold_{fold_num+1}': pred2_train}, index=X_train.index)
            df_pred3_train = pd.DataFrame({f'xgb_fold_{fold_num+1}': pred3_train}, index=X_train.index)

            # Collect true labels
            df_true_test = pd.DataFrame({'true_value': y_test}, index=X_test.index)
            df_true_train = pd.DataFrame({'true_value': y_train}, index=X_train.index)

            # Concatenate predictions and true labels for test set
            df_fold_test = pd.concat([df_true_test, df_pred1_test, df_pred2_test, df_pred3_test], axis=1)

            # Concatenate predictions and true labels for train set
            df_fold_train = pd.concat([df_true_train, df_pred1_train, df_pred2_train, df_pred3_train], axis=1)

            predictions_list_test.append(df_fold_test)
            predictions_list_train.append(df_fold_train)

        else:
            raise ValueError("Invalid model_type. Options are 'classification' or 'regression'.")

    return predictions_list_train, predictions_list_test, models_dict


def calculate_uncertainty_classification(predictions_df, num_classes):
   """
   Calculates uncertainty for classification tasks using the entropy of the average predicted probabilities across an ensemble of models.

   Parameters:
   - predictions_df (pd.DataFrame): DataFrame with columns for each model's predicted probabilities for each sample.
   Each column contains a list of probabilities for all classes.
   - num_classes (int): Number of classes in the classification task.

   Returns:
   - uncertainties (pd.Series): Series with uncertainty values (entropy) for each sample.
   """
   logger.info(f"Calculating uncertainty for classification task.")
   # Initialize an array to store the average probabilities for each sample and class
   avg_probs = np.zeros((len(predictions_df), num_classes))

   # Calculate the average probability for each class across all models
   for i, row in enumerate(predictions_df.iterrows()):
      # Initialize an array to store the sum of probabilities for each class across models
      sum_probs = np.zeros(num_classes)

      # Iterate over each model's predicted probabilities (each column in the DataFrame)
      for model_col in predictions_df.columns:
         class_probs = np.array(row[1][model_col])  # Extract the list of class probabilities
         sum_probs += class_probs  # Sum the probabilities for each class

      # Average the probabilities by dividing by the number of models
      avg_probs[i] = sum_probs / len(predictions_df.columns)

   # Calculate the entropy for each sample using the average probabilities
   uncertainties = entropy(avg_probs, axis=1)  # Entropy expects shape (n_samples, n_classes)

   # Create a Series with the uncertainty values, aligned with the DataFrame's index
   uncertainties_series = pd.Series(uncertainties, index=predictions_df.index)

   return uncertainties_series


def calculate_uncertainty_regression(predictions_df):
   """
   Calculates uncertainty for regression tasks using variance of predictions.

   Parameters:
   - predictions_df (pd.DataFrame): DataFrame with true values and predictions from models.

   Returns:
   - uncertainties (pd.Series): Series with uncertainty values (variance) for each sample.
   """
   logger.info(f"Calculating uncertainty for regression task.")
   # Get prediction columns (assuming model predictions contain 'model_' in their names)
   pred_cols = [col for col in predictions_df.columns if re.search(r'model_\d+', col)]
   if not pred_cols:
      raise ValueError("No prediction columns found for regression uncertainty calculation.")

   # For each sample, compute variance of predictions
   uncertainties = predictions_df[pred_cols].var(axis=1)

   return uncertainties


def compute_objective(obj_fn_str, predictions_dict, target_types, num_classes_dict):
    """
    Computes the objective function for each sample based on target-specific predictions and weights.

    Parameters:
    - obj_fn_str (str): Objective function as a string expression involving target variables.
                        Example: '(structure_type_class_2 + 0.2 * optical_absorption) / (0.1 * particle_size)'
    - predictions_dict (dict): Dictionary where keys are target names and values are prediction DataFrames.
                                - For classification targets: Each DataFrame contains columns for each model,
                                  and each cell contains a list of probabilities for all classes.
                                - For regression targets: Each DataFrame contains columns for each model's prediction.
    - target_types (dict): Dictionary mapping target names to their types ('classification' or 'regression').
    - num_classes_dict (dict): Dictionary mapping classification target names to their number of classes.
                               Example: {'structure_type': 3}

    Returns:
    - objectives (pd.Series): Series with objective values for each sample, indexed by sample identifiers.
    """
    # Initialize an empty DataFrame to hold weighted predictions for all targets
    aggregated_predictions = pd.DataFrame(index=next(iter(predictions_dict.values())).index)

    # Dictionary to map variable names to column names in aggregated_predictions
    variable_column_mapping = {}

    for target, pred_df in predictions_dict.items():
       
        if 'classification' in target_types[target]:
            # Number of classes for this classification target
            num_classes = len(pred_df.iloc[0, 0]) 
            if not num_classes:
                raise ValueError(f"Number of classes not specified for classification target '{target}'.")

            # Calculate average probabilities across models
            num_models = len(pred_df.columns)
            sum_probs = np.zeros((len(pred_df), num_classes))

            for model_col in pred_df.columns:
                # Convert list of probabilities to numpy array
                model_probs = pred_df[model_col].apply(lambda x: np.array(x))
                # Stack into a 2D array: (num_samples, num_classes)
                model_probs_matrix = np.stack(model_probs.values)
                sum_probs += model_probs_matrix

            # Average probabilities
            avg_probs = sum_probs / num_models  # Shape: (num_samples, num_classes)

            # Add each class's average probability as a separate column, weighted
            for class_idx in range(num_classes):
                col_name = f"{target}_class_{class_idx}"
                variable_name = col_name  # Variable name used in obj_fn_str
                aggregated_predictions[col_name] = avg_probs[:, class_idx]
                # Map variable name to column name
                variable_column_mapping[variable_name] = col_name

        elif 'regression' in target_types[target]:
            # Calculate average predictions across models
            avg_pred = pred_df.mean(axis=1)
            col_name = f"{target}"
            variable_name = col_name  # Variable name used in obj_fn_str
            aggregated_predictions[col_name] = avg_pred
            # Map variable name to column name
            variable_column_mapping[variable_name] = col_name
        else:
            raise ValueError(f"Unknown target type for '{target}'. Expected 'classification' or 'regression'.")

    # Extract variable names from obj_fn_str using regular expressions
    # Match variable names: alphanumeric characters, underscores, and possibly class indices
    variable_names_in_expr = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', obj_fn_str))

    # Ensure all variables in obj_fn_str are available in variable_column_mapping
    for var_name in variable_names_in_expr:
        if var_name not in variable_column_mapping:
            raise ValueError(f"Variable '{var_name}' in obj_fn_str is not available in the predictions.")

    # Prepare a dictionary to map variable names to column data
    # This will be used to evaluate the expression for each sample
    variables_data = {var_name: aggregated_predictions[variable_column_mapping[var_name]] for var_name in variable_names_in_expr}

    # Evaluate the expression for each sample
    objectives = ne.evaluate(obj_fn_str, local_dict=variables_data)

    # Create a pandas Series with the objectives, indexed by sample identifiers
    objectives_series = pd.Series(objectives, index=aggregated_predictions.index)

    return objectives_series


def compute_acquisition(uncertainty, objective, entropy_gain, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Computes the acquisition function combining uncertainty, objective, and entropy gain.

    Parameters:
    uncertainty (pd.Series): Uncertainty values for each sample.
    objective (pd.Series): Objective values for each sample.
    entropy_gain (pd.Series): Entropy gain values for each sample.
    alpha (float): Weight for objective.
    beta (float): Weight for uncertainty.
    gamma (float): Weight for entropy gain.

    Returns:
    acquisition (pd.Series): Acquisition values for each sample.
    """
    # Normalize uncertainty, objective, and entropy_gain to [0, 1]
    uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) if uncertainty.max() != uncertainty.min() else 0.0
    objective_norm = (objective - objective.min()) / (objective.max() - objective.min()) if objective.max() != objective.min() else 0.0
    entropy_gain_norm = (entropy_gain - entropy_gain.min()) / (entropy_gain.max() - entropy_gain.min()) if entropy_gain.max() != entropy_gain.min() else 0.0

    acquisition = alpha * objective_norm + beta * uncertainty_norm + gamma * entropy_gain_norm

    return acquisition

def predict_with_models(X, models, model_type='classification'):
   """
   Makes predictions on X using the list of models.

   Parameters:
   - X (pd.DataFrame): Feature data.
   - models (list): List of trained models.
   - model_type (str): 'classification' or 'regression'.

   Returns:
   - predictions_df (pd.DataFrame): DataFrame with predictions.
   """
   predictions_list = []
   for i, model in enumerate(models):
      if model_type == 'classification':
         prob = model.predict_proba(X)
         df_pred = pd.DataFrame({f'model_{i+1}': [list(p) for p in prob]}, index=X.index)
      elif model_type == 'regression':
         pred = model.predict(X)
         df_pred = pd.DataFrame({f'model_{i+1}': pred}, index=X.index)
      else:
         raise ValueError("Invalid model_type. Options are 'classification' or 'regression'.")

      predictions_list.append(df_pred)

   # Concatenate predictions
   predictions_df = pd.concat(predictions_list, axis=1)

   return predictions_df

def save_unique_figure(fig, base_filename, formats=('png', 'svg')):
    """
    Saves the figure to the current directory with a unique filename if a file already exists.

    Parameters:
    - fig: The matplotlib figure object.
    - base_filename (str): The base filename for the figure.
    - formats (tuple): File formats to save the figure (e.g., 'png', 'svg').

    Returns:
    - None
    """
    for fmt in formats:
        filename = f"{base_filename}.{fmt}"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_filename}_{counter}.{fmt}"
            counter += 1
        if fmt == 'svg':
            # Ensure formattable text in SVG
            fig.savefig(filename, format=fmt, metadata={'Title': 'Performance Metrics'})
        else:
            fig.savefig(filename, format=fmt)

import os
import matplotlib.pyplot as plt

def save_unique_figure(fig, base_filename, formats=('png', 'svg')):
    """
    Saves the figure to the current directory with a unique filename if a file already exists.

    Parameters:
    - fig: The matplotlib figure object.
    - base_filename (str): The base filename for the figure.
    - formats (tuple): File formats to save the figure (e.g., 'png', 'svg').

    Returns:
    - None
    """
    # Define a unique base name for all formats
    base_filename_unique = base_filename
    counter = 1
    while any(os.path.exists(f"{base_filename_unique}.{fmt}") for fmt in formats):
        base_filename_unique = f"{base_filename}_{counter}"
        counter += 1

    # Save the figure in each specified format
    for fmt in formats:
        filename = f"{base_filename_unique}.{fmt}"
        if fmt == 'svg':
            # Ensure formattable text in SVG
            fig.savefig(filename, format=fmt, metadata={'Title': 'Performance Metrics'})
        else:
            fig.savefig(filename, format=fmt)

def plot_performance(performance_metrics, target_types):
    """
    Plots the performance metrics over iterations for each target, saving as a single figure
    with classification targets in one subplot and regression targets in their own subplots.

    Parameters:
    - performance_metrics (list): List of dictionaries containing 'iteration', 'selected_models', and 'metrics'.
    - target_types (dict): Dictionary mapping target names to their types ('classification' or 'regression').

    Returns:
    - None
    """
    # Initialize dictionary to hold metrics per target
    target_to_metrics = {target: [] for target in target_types.keys()}

    # Iterate over each iteration's metrics
    for metric_entry in performance_metrics:
        metrics = metric_entry['metrics']
        for target, metric_value in metrics.items():
            target_to_metrics[target].append(metric_value)

    # Separate targets by type
    classification_targets = [t for t, t_type in target_types.items() if t_type == 'classification']
    regression_targets = [t for t, t_type in target_types.items() if t_type == 'regression']

    num_classification = len(classification_targets)
    num_regression = len(regression_targets)

    # Total number of plots will be 1 for classification and 1 per regression target
    total_plots = int(num_classification > 0) + num_regression

    if total_plots == 0:
        logger.warning("No targets to plot.")
        return

    fig, axes = plt.subplots(nrows=total_plots, figsize=(12, 6 * total_plots))

    # Handle the case when there's only one subplot (axes will not be an array)
    if total_plots == 1:
        axes = [axes]

    plot_idx = 0
    plt.rcParams['svg.fonttype'] = 'none'
    # Plot Classification Metrics
    if classification_targets:
        ax = axes[plot_idx]
        for target in classification_targets:
            iterations = range(1, len(target_to_metrics[target]) + 1)
            ax.plot(iterations, target_to_metrics[target], marker='o', label=target)
        ax.set_title('Classification Targets Performance over Iterations', fontsize=14)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Plot Regression Metrics, each in its own subplot
    for target in regression_targets:
        ax = axes[plot_idx]
        iterations = range(1, len(target_to_metrics[target]) + 1)
        ax.plot(iterations, target_to_metrics[target], marker='s', label=target)
        ax.set_title(f'Regression Target {target} Performance over Iterations', fontsize=14)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    plt.tight_layout()
    save_unique_figure(fig, 'performance_metrics_combined')
    plt.show()



def generate_sampling_grid(X, num_grid_points_per_num_var=10): 
   """
    Generates a sampling grid based on the ranges of each feature in X.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - num_grid_points_per_num_var (int or dict, optional): 
        - If int, the same number of grid points is used for all numerical variables.
        - If dict, specify the number of grid points per numerical variable.

    Returns:
    - grid_df (pd.DataFrame): DataFrame representing the sampling grid.
   """
   grid_dict = {}
   for col in X.columns:
      if X[col].dtype == 'object' or X[col].nunique() <= 2:
         # Categorical variable: include all unique categories
         grid_dict[col] = X[col].unique()
      else:
         # Numerical variable: handle custom grid points if provided
         if isinstance(num_grid_points_per_num_var, dict):
            if col in num_grid_points_per_num_var:
               if num_grid_points_per_num_var[col] == 'unique':
                  grid_dict[col] = X[col].unique()
               else:
                  grid_dict[col] = num_grid_points_per_num_var[col]
            else:
               num_points = num_grid_points_per_num_var.get(col, 10)  # Default to 10 if not specified
               min_val = X[col].min()
               max_val = X[col].max()
               if min_val == max_val:
                  grid_dict[col] = [min_val]
               else:
                  grid_dict[col] = np.linspace(min_val, max_val, num=num_points)
         else:
            num_points = num_grid_points_per_num_var
            min_val = X[col].min()
            max_val = X[col].max()
            if min_val == max_val:
               grid_dict[col] = [min_val]
            else:
               grid_dict[col] = np.linspace(min_val, max_val, num=num_points)
        # Print all possible values for categorical variables and numerical ranges for numerical variables
   for col, values in grid_dict.items():
      logger.info(f"Feature '{col}': {values}")
   # Create cartesian product of all feature grids
   grid_df = pd.DataFrame(np.array(np.meshgrid(*grid_dict.values())).T.reshape(-1, len(grid_dict)), columns=grid_dict.keys())
   # Convert data types back to original
   for col in X.columns:
      if X[col].dtype == 'object':
         grid_df[col] = grid_df[col].astype(X[col].dtype)
   logger.info(f"Generated sampling grid with {len(grid_df)} points.")
   logger.info(f"Grid columns: {grid_df.columns}")
   logger.info(f"Grid head:\n{grid_df.head()}")
   return grid_df

def apply_constraints_to_grid(grid_df, variable_constraints):
    """
    Applies constraints to the grid:
    1. Ensures that only one of the one-hot encoded columns can be 'hot' for each categorical variable.
    2. Applies custom variable constraints provided by the user, with optional enforcement of mutual constraints.
    
    Parameters:
    - grid_df (pd.DataFrame): The sampling grid.
    - variable_constraints (list of dicts): List of constraints where each constraint is a dictionary
      with 'conditions', 'assignments', and optionally 'mutual_constraint' keys.
    
    Returns:
    - grid_df (pd.DataFrame): The constrained sampling grid.
    """
    
    # Constraint: Eliminate rows that violate variable_constraints
    if variable_constraints is not None:
        for constraint in variable_constraints:
            conditions = constraint.get('conditions', {})
            assignments = constraint.get('assignments', {})
            mutual_constraint = constraint.get('mutual_constraint', False)  # Default is False

            # Build a boolean mask for the conditions (True case)
            condition_mask = pd.Series(True, index=grid_df.index)
            for col, val in conditions.items():
                if col in grid_df.columns:
                    condition_mask &= grid_df[col] == val
                else:
                    logger.warning(f"Condition column '{col}' not found in grid_df.")

            # Build a boolean mask for the assignments (True case)
            assignment_mask = pd.Series(True, index=grid_df.index)
            for col, val in assignments.items():
                if col in grid_df.columns:
                    assignment_mask &= grid_df[col] == val
                else:
                    logger.warning(f"Assignment column '{col}' not found in grid_df.")

            # Apply mutual constraint if specified
            if mutual_constraint:
                # Build a boolean mask for the reverse condition (False case)
                reverse_condition_mask = pd.Series(True, index=grid_df.index)
                for col, val in conditions.items():
                    if col in grid_df.columns:
                        reverse_condition_mask &= grid_df[col] != val
                    else:
                        logger.warning(f"Reverse condition column '{col}' not found in grid_df.")

                # Build a boolean mask for the reverse assignments (False case)
                reverse_assignment_mask = pd.Series(True, index=grid_df.index)
                for col, val in assignments.items():
                    if col in grid_df.columns:
                        reverse_assignment_mask &= grid_df[col] != val
                    else:
                        logger.warning(f"Reverse assignment column '{col}' not found in grid_df.")

                # Identify rows that meet the reverse condition but violate the reverse assignments
                reverse_violation_mask = (reverse_condition_mask & (~reverse_assignment_mask))

            else:
                reverse_violation_mask = pd.Series(False, index=grid_df.index)  # No reverse violations

            # Identify rows that meet the conditions but violate the assignments (True case)
            violation_mask = (condition_mask & (~assignment_mask))

            # Combine both violation masks
            combined_violation_mask = violation_mask | reverse_violation_mask

            # Remove the violating rows
            num_violations = combined_violation_mask.sum()
            if num_violations > 0:
                logger.info(f"Removing {num_violations} grid points that violate the constraints: {constraint}")
                grid_df = grid_df[~combined_violation_mask]
            else:
                logger.info(f"No violations found for constraint: {constraint}, is the constraint correct?")
    logger.info(f"Applied constraints to the grid. New grid size: {len(grid_df)} points.")
    return grid_df


def active_learning_loop(
   X,
   y_dict,
   target_types,
   obj_fn_str, # Objective function as a string expression
   categorical_cols=None,
   num_classes_dict=None,
   target_weights=None,
   initial_train_size=None,  # Default to total size if None
   iterations=10,
   batch_size=8,  # Default to 8 samples per iteration
   alpha=1.0, # Weight for objective in acquisition function
   beta=1.0, # Weight for uncertainty in acquisition function
   gamma=1.0,  # Weight for sampling entropy
   num_grid_points_per_num_var=10,  # Default number of grid points for numerical variables
   user_num_grid_points=None,  # Optional dict to override grid points per numerical variable
   variable_constraints=None,  # Optional constraints on variables that reduce grid points
):
    """
    Performs active learning loop.

    Parameters:
    - X (pd.DataFrame): The feature dataframe.
    - y_dict (dict): Dictionary of target variables.
    - target_types (dict): Dictionary mapping target names to their types ('classification' or 'regression').
    - target_weights (dict, optional): Dictionary mapping target names to their weights in the objective function.
    - initial_train_size (int, optional): Number of initial training samples. Defaults to total size.
    - iterations (int): Number of active learning iterations.
    - batch_size (int): Number of samples to select in each iteration.
    - alpha (float): Weight for uncertainty in acquisition function.
    - beta (float): Weight for objective in acquisition function.
    - num_grid_points_per_num_var (int or dict, optional): 
        - If int, the same number of grid points is used for all numerical variables.
        - If dict, specify the number of grid points per numerical variable.
    - user_num_grid_points (dict, optional): Override the number of grid points per numerical variable.
    - obj_fn_str (str): Objective function as a string expression involving target variables.
    - categorical_cols (list, optional): List of categorical columns in X.
    - num_classes_dict (dict, optional): Dictionary mapping classification target names to their number of classes.
    - variable_constraints (dict, optional): Dictionary of constraints on variables that reduce grid points.

    Returns:
    - None
    """
    if target_weights is None:
        target_weights = {target: 1.0 for target in y_dict.keys()}

    # Generate sampling grid
    if user_num_grid_points is not None:
        grid_points = user_num_grid_points
    else:
        grid_points = num_grid_points_per_num_var

    X_grid = generate_sampling_grid(X, num_grid_points_per_num_var=grid_points)
    # Apply constraints to the grid
    X_grid = apply_constraints_to_grid(X_grid, variable_constraints)

    # Encode categorical variables
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols)
        logger.info("\nData after encoding categorical variables:")
        logger.info(X.head())
        # also for the grid
        X_grid = pd.get_dummies(X_grid, columns=categorical_cols)
        # if there are missing columns in the grid, fill with 0
        missing_cols = set(X.columns) - set(X_grid.columns)
        for col in missing_cols:
            X_grid[col] = 0
        # we need to impose same dtypes for X and X_grid
        X_grid = X_grid[X.columns]
        X_grid = X_grid.astype(X.dtypes)

    
    # Initialize training and pool data
    # Default initial_train_size is total size of samples
    if initial_train_size is None:
        initial_train_size = len(X)
    else:
        initial_train_size = initial_train_size

    if initial_train_size < len(X):
        # Sample initial_train_size as X_train
        initial_train_indices = y_dict[next(iter(y_dict))].sample(n=initial_train_size, random_state=42).index.tolist()

        X_train = X.loc[initial_train_indices].copy()
        y_train_dict = {target: y.loc[initial_train_indices].copy() for target, y in y_dict.items()}

        # Assign the remaining samples as X_test
        X_test = X.drop(index=initial_train_indices).copy()
        y_test_dict = {target: y.drop(index=initial_train_indices).copy() for target, y in y_dict.items()}

        # Define X_pool as the combination of X_grid and X_test

        # Remove any samples in X_train from X_pool to ensure no overlap
        X_pool = pd.concat([X_grid, X_test], ignore_index=True).drop_duplicates().reset_index(drop=True)
        # Remove any samples from X_pool that are present in X_train
        X_pool = X_pool[~X_pool.apply(tuple, axis=1).isin(X_train.apply(tuple, axis=1))]

        # Initialize y_pool_dict:
        # - For samples from X_test: labels are known
        # - For samples from X_grid: labels are unknown (NaN)
        y_pool_dict = {}
        for target in y_dict.keys():
            y_pool_test = y_test_dict[target].reset_index(drop=True)
            y_pool_grid = pd.Series([np.nan] * len(X_grid), name=target)
            y_pool_combined = pd.concat([y_pool_test, y_pool_grid], ignore_index=True)
            y_pool_combined = y_pool_combined[~X_pool.apply(tuple, axis=1).isin(X_train.apply(tuple, axis=1))]
            y_pool_dict[target] = y_pool_combined
    else:
        # initial_train_size >= len(X), assign all data to X_train
        X_train = X.copy()
        y_train_dict = {target: y.copy() for target, y in y_dict.items()}

        
       # Remove any samples in X_train from X_pool to ensure no overlap
        # Define X_pool as the grid only
        X_pool = X_grid.copy()
        # Any missing columns in X_pool will be filled with 0
        # Remove any samples from X_pool that are present in X_train
        X_pool = X_pool[~X_pool.apply(tuple, axis=1).isin(X_train.apply(tuple, axis=1))]
        # Initialize y_pool_dict with NaNs for all grid samples
        y_pool_dict = {target: pd.Series([np.nan] * len(X_pool), index=X_pool.index) for target in y_dict.keys()}

    # Log the initial setup
    logger.info(f"Initial training set size: {len(X_train)}")
    if initial_train_size < len(X):
        logger.info(f"Initial testing set size: {len(X_test)}")
    logger.info(f"Pool size after removing overlaps with X_train: {len(X_pool)}")
    # For logging performance metrics
    performance_metrics = []

    # Dictionary to keep track of the selected model type per target
    selected_model_types = {target: None for target in y_dict.keys()}
    retained_models = {target: [] for target in y_dict.keys()}
    
    for iteration in range(iterations):
        logger.info(f"\n--- Iteration {iteration + 1} ---")

        models_dict = {}
        predictions_train_dict = {}
        predictions_test_dict = {}
        splits_dict = {}
        model_performance = {}

        # Train models for each target
        for target, y_train in y_train_dict.items():
            model_type = target_types[target]
            logger.info(f"Training models for target '{target}' ({model_type})")

            # Determine if StratifiedKFold is needed
            if model_type == 'classification':
                splits = get_unique_kfold_splits(X_train, n_splits=5, n_repeats=3, y_cat=y_train)
            else:
                splits = get_unique_kfold_splits(X_train, n_splits=5, n_repeats=3)

            splits_dict[target] = splits

            predictions_train, predictions_test, models = train_models_and_collect_predictions(
                X_train, y_train, model_type=model_type, splits=splits
            )

            predictions_train_dict[target] = predictions_train
            predictions_test_dict[target] = predictions_test
            models_dict[target] = models

            # Calculate average metrics across folds for each model
            avg_metrics_test = calculate_average_metrics(predictions_test, model_type=model_type)
            logger.info(f"Average test metrics for target '{target}':\n{avg_metrics_test}")

            # Select the best model type based on average performance
            if model_type == 'classification':
                # Select model with highest F1-score
                best_row = avg_metrics_test.loc[avg_metrics_test['f1_score'].idxmax()]
                # register the best metric
                model_performance[target] = best_row['f1_score']
            elif model_type == 'regression':
                # Select model with highest R2
                best_row = avg_metrics_test.loc[avg_metrics_test['rmse'].idxmin()]
                # register the best metric
                model_performance[target] = best_row['rmse']
            else:
                raise ValueError(f"Unknown model type '{model_type}' for target '{target}'.")

            best_model_type = best_row['model']
            selected_model_types[target] = best_model_type
            logger.info(f"Selected best model type for target '{target}': {best_model_type}")

            # Retain only the models of the selected type
            retained_models[target] = models[best_model_type]
            logger.info(f"Retained {len(retained_models[target])} models of type '{best_model_type}' for target '{target}'")


        # Make predictions on pool data using the retained models
        predictions_pool_dict = {}
        uncertainty_dict = {}
        for target, models in models_dict.items():
            model_type = target_types[target]
            logger.info(f"Predicting on pool data for target '{target}' ({model_type})")

            if len(models) == 0:
                logger.warning(f"No models available for target '{target}'. Skipping.")
                continue

            predictions_pool = predict_with_models(X_pool, retained_models[target], model_type=model_type)
            predictions_pool_dict[target] = predictions_pool

            # Calculate uncertainty
            if model_type == 'classification':
                # Determine number of classes
                num_classes = len(predictions_pool.iloc[0, 0])
                uncertainty = calculate_uncertainty_classification(predictions_pool, num_classes)
            elif model_type == 'regression':
                uncertainty = calculate_uncertainty_regression(predictions_pool)
            else:
                raise ValueError(f"Unknown model type '{model_type}' for target '{target}'.")

            uncertainty_dict[target] = uncertainty
        
        # Normalize uncertainties to [0, 1] range
        for target, uncertainty in uncertainty_dict.items():
            uncertainty_dict[target] = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) if uncertainty.max() != uncertainty.min() else uncertainty
        
        # Combine uncertainties weighted by target_weights
        total_uncertainty = pd.Series(0, index=X_pool.index)
        for target, uncertainty in uncertainty_dict.items():
            total_uncertainty += target_weights[target]['unc'] * uncertainty


        objective = compute_objective(
                    obj_fn_str,
                    predictions_pool_dict,
                    target_types,
                    num_classes_dict
                )
        
        # Initialize selected indices for this iteration
        selected_indices = []
        ### Entropy gain and acquisition cycle ###
        logger.info(f"Calculating scaler to normalize X values for entropy gain function.")
        scaler = StandardScaler()
        # we apply in the concatenated X_train and X_pool
        X_total = pd.concat([X_train, X_pool])
        scaler.fit(X_total)
        del(X_total)         # free memory
        # Normalize X_train and X_pool while keeping original indices
        X_pool_scaled = pd.DataFrame(scaler.transform(X_pool), index=X_pool.index, columns=X_pool.columns)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        for _ in range(batch_size):
            # Fit NearestNeighbors on the current training set
            nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(X_train_scaled.values)
            
            # Compute distances for all pool samples at once
            distances, _ = nbrs.kneighbors(X_pool_scaled.values)
            
            # Calculate average distances for each pool sample
            avg_distances = distances.mean(axis=1)
            
            # Create a pandas Series for entropy_gain
            entropy_gain = pd.Series(avg_distances, index=X_pool_scaled.index)
            
            # Compute acquisition function
            acquisition = compute_acquisition(total_uncertainty, objective, entropy_gain, alpha=alpha, beta=beta, gamma=gamma)
            
            # Select the sample with the highest acquisition score
            selected_idx = acquisition.idxmax()
            selected_indices.append(selected_idx)
            
            # Update the training set and pool
            X_train_scaled = pd.concat([X_train_scaled, X_pool_scaled.loc[[selected_idx]]])
            X_pool_scaled = X_pool_scaled.drop(index=selected_idx)
        # free memory
        del(scaler)
        del(X_train_scaled)
        del(X_pool_scaled)
        logger.info(f"Selected samples: {selected_indices}")

        # Retrieve the selected samples from the grid
        X_selected = X_pool.loc[selected_indices].copy()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(f"Selected samples coordinates:\n{X_selected}")

        #### SIMULATE LABELLING #### 
        # Simulate labeling: For each target, pick a model and predict y_true
        y_true_selected = {}
        selected_models_info = {}

        for target in y_dict.keys():
            models = models_dict[target]
            if len(models) == 0:
                logger.warning(f"No models available for target '{target}'. Skipping labeling.")
                y_true_selected[target] = pd.Series([np.nan] * len(X_selected), index=X_selected.index)
                selected_models_info[target] = [None] * len(X_selected)
                continue

            # Randomly select a model by idx
            idx_selected = np.random.choice(len(retained_models[target]))
            selected_model = retained_models[target][idx_selected]
            selected_models_info[target] = selected_model

            # Predict y_true using the selected models
            if target_types[target] == 'classification':
                y_pred = [selected_model.predict(X_selected.values)]
            elif target_types[target] == 'regression':
                y_pred = [selected_model.predict(X_selected.values)]
            else:
                y_pred = [np.nan] * len(X_selected)

            y_true_selected[target] = pd.Series(y_pred[0], index=X_selected.index)

            # Log the sample coordinates and which model idx was used for each sample
            logger.info(f"Selected models for target '{target}': {idx_selected}")
            logger.info(f"Predicted labels for target '{target}':\n{y_true_selected[target]}")
        
        # ensure same data types in each column
        X_selected = X_selected.astype(X_train.dtypes)    
        # Add selected samples and their labels to training data
        X_train = pd.concat([X_train, X_selected])
        
        logger.info(f"Updated training set size: {len(X_train)}")
        logger.info(f"Updated pool size: {len(X_pool) - len(X_selected)}")
        # logger.info(f"Head of updated training set:\n{X_train.head()}")
        # logger.info(f"Columns of updated training set:\n{X_train.columns}")

        for target in y_dict.keys():
            y_true = y_true_selected[target]
            y_train_dict[target] = pd.concat([y_train_dict[target], y_true])

        # Remove selected samples from pool
        X_pool = X_pool.drop(index=selected_indices)
        for target in y_dict.keys():
            y_pool_dict[target] = y_pool_dict[target].drop(index=selected_indices)

        # Collect performance metrics (Placeholder: Extend this section as needed)
        # Here, you can evaluate model performance on a validation set or via cross-validation
        # For demonstration, we'll log the selected models
        performance_metrics.append({
            'iteration': iteration + 1,
            'selected_models': selected_models_info,
            'metrics': model_performance,
        })

    # After iterations, plot performance metrics
    if performance_metrics:
        plot_performance(performance_metrics, target_types)
    else:
        logger.info("No performance metrics to plot.")


def train_models_and_collect_predictions(X, y, model_type='classification', splits=None):
    """
    Trains models and collects predictions.

    Parameters:
    - X (pd.DataFrame): Feature data.
    - y (pd.Series): Target variable.
    - model_type (str): 'classification' or 'regression'.
    - splits (list): List of (train_index, test_index) splits.

    Returns:
    - predictions_list_train (list): List of DataFrames with train predictions per fold.
    - predictions_list_test (list): List of DataFrames with test predictions per fold.
    - models_dict (dict): Dictionary mapping model types to their trained models.
    """
    if splits is None:
        # Default to KFold with 5 splits
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X))

    # Initialize models_dict based on model_type
    models_dict = {'LR': [], 'RF': [], 'XGB': []}

    predictions_list_test = []
    predictions_list_train = []

    for fold_num, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_type == 'classification':
            # Models: Logistic Regression, Random Forest Classifier, XGBoost Classifier
            model1 = LogisticRegression(max_iter=1000, random_state=42)
            model2 = RandomForestClassifier(random_state=42)
            model3 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

            model1.fit(X_train, y_train)
            model2.fit(X_train, y_train)
            model3.fit(X_train, y_train)

            # Store models
            models_dict['LR'].append(model1)
            models_dict['RF'].append(model2)
            models_dict['XGB'].append(model3)

            # Collect predictions
            prob1_test = model1.predict_proba(X_test)
            prob2_test = model2.predict_proba(X_test)
            prob3_test = model3.predict_proba(X_test)

            prob1_train = model1.predict_proba(X_train)
            prob2_train = model2.predict_proba(X_train)
            prob3_train = model3.predict_proba(X_train)

            # For each model, create a DataFrame of predictions for test set
            df_pred1_test = pd.DataFrame({f'lr_fold_{fold_num+1}_proba': prob1_test.tolist()}, index=X_test.index)
            df_pred2_test = pd.DataFrame({f'rf_fold_{fold_num+1}_proba': prob2_test.tolist()}, index=X_test.index)
            df_pred3_test = pd.DataFrame({f'xgb_fold_{fold_num+1}_proba': prob3_test.tolist()}, index=X_test.index)

            # For each model, create a DataFrame of predictions for train set
            df_pred1_train = pd.DataFrame({f'lr_fold_{fold_num+1}_proba': prob1_train.tolist()}, index=X_train.index)
            df_pred2_train = pd.DataFrame({f'rf_fold_{fold_num+1}_proba': prob2_train.tolist()}, index=X_train.index)
            df_pred3_train = pd.DataFrame({f'xgb_fold_{fold_num+1}_proba': prob3_train.tolist()}, index=X_train.index)

            # Collect true labels
            df_true_test = pd.DataFrame({'true_label': y_test}, index=X_test.index)
            df_true_train = pd.DataFrame({'true_label': y_train}, index=X_train.index)

            # Concatenate predictions and true labels for test set
            df_fold_test = pd.concat([df_true_test, df_pred1_test, df_pred2_test, df_pred3_test], axis=1)

            # Concatenate predictions and true labels for train set
            df_fold_train = pd.concat([df_true_train, df_pred1_train, df_pred2_train, df_pred3_train], axis=1)

            predictions_list_test.append(df_fold_test)
            predictions_list_train.append(df_fold_train)

        elif model_type == 'regression':
            # Models: Linear Regression, Random Forest Regressor, XGBoost Regressor
            model1 = LinearRegression()
            model2 = RandomForestRegressor(random_state=42)
            model3 = xgb.XGBRegressor(random_state=42)

            model1.fit(X_train, y_train)
            model2.fit(X_train, y_train)
            model3.fit(X_train, y_train)

            # Store models
            models_dict['LR'].append(model1)
            models_dict['RF'].append(model2)
            models_dict['XGB'].append(model3)

            # Collect predictions for test set
            pred1_test = model1.predict(X_test)
            pred2_test = model2.predict(X_test)
            pred3_test = model3.predict(X_test)

            # Collect predictions for train set
            pred1_train = model1.predict(X_train)
            pred2_train = model2.predict(X_train)
            pred3_train = model3.predict(X_train)

            # For each model, create a DataFrame of predictions for test set
            df_pred1_test = pd.DataFrame({f'lr_fold_{fold_num+1}': pred1_test}, index=X_test.index)
            df_pred2_test = pd.DataFrame({f'rf_fold_{fold_num+1}': pred2_test}, index=X_test.index)
            df_pred3_test = pd.DataFrame({f'xgb_fold_{fold_num+1}': pred3_test}, index=X_test.index)

            # For each model, create a DataFrame of predictions for train set
            df_pred1_train = pd.DataFrame({f'lr_fold_{fold_num+1}': pred1_train}, index=X_train.index)
            df_pred2_train = pd.DataFrame({f'rf_fold_{fold_num+1}': pred2_train}, index=X_train.index)
            df_pred3_train = pd.DataFrame({f'xgb_fold_{fold_num+1}': pred3_train}, index=X_train.index)

            # Collect true labels
            df_true_test = pd.DataFrame({'true_value': y_test}, index=X_test.index)
            df_true_train = pd.DataFrame({'true_value': y_train}, index=X_train.index)

            # Concatenate predictions and true labels for test set
            df_fold_test = pd.concat([df_true_test, df_pred1_test, df_pred2_test, df_pred3_test], axis=1)

            # Concatenate predictions and true labels for train set
            df_fold_train = pd.concat([df_true_train, df_pred1_train, df_pred2_train, df_pred3_train], axis=1)

            predictions_list_test.append(df_fold_test)
            predictions_list_train.append(df_fold_train)
        else:
            raise ValueError("Invalid model_type. Options are 'classification' or 'regression'.")

    return predictions_list_train, predictions_list_test, models_dict


def calculate_average_metrics(predictions_list, model_type='classification'):
    """
    Calculates average metrics per model type across all folds.

    Parameters:
    - predictions_list (list): List of DataFrames with predictions per fold.
    - model_type (str): 'classification' or 'regression'.

    Returns:
    - avg_metrics_df (pd.DataFrame): DataFrame with average metrics per model type.
    """
    metrics = []
    for fold_predictions in predictions_list:
        if model_type == 'classification':
            true_labels = fold_predictions['true_label']
            # Aggregate predictions by model type
            model_types = ['lr', 'rf', 'xgb']
            for model_type_short in model_types:
                prob_cols = [col for col in fold_predictions.columns if col.startswith(model_type_short)]
                if not prob_cols:
                    continue
                # Average probabilities across all folds/models of the same type
                probs = np.mean([fold_predictions[col].tolist() for col in prob_cols], axis=0)
                predicted_labels = np.argmax(probs, axis=1)
                accuracy = accuracy_score(true_labels, predicted_labels)
                f1 = f1_score(true_labels, predicted_labels, average='weighted')
                precision = precision_score(true_labels, predicted_labels, average='weighted')
                recall = recall_score(true_labels, predicted_labels, average='weighted')
                metrics.append({
                    'model': model_type_short.upper(),
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall
                })
        elif model_type == 'regression':
            true_values = fold_predictions['true_value']
            # Aggregate predictions by model type
            model_types = ['lr', 'rf', 'xgb']
            for model_type_short in model_types:
                pred_cols = [col for col in fold_predictions.columns if col.startswith(model_type_short)]
                if not pred_cols:
                    continue
                # Average predictions across all folds/models of the same type
                preds = np.mean([fold_predictions[col] for col in pred_cols], axis=0)
                rmse = root_mean_squared_error(true_values, preds)
                mae = mean_absolute_error(true_values, preds)
                r2 = r2_score(true_values, preds)
                metrics.append({
                    'model': model_type_short.upper(),
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })
    avg_metrics_df = pd.DataFrame(metrics)
    if not avg_metrics_df.empty:
        avg_metrics_df = avg_metrics_df.groupby('model').mean().reset_index()
    return avg_metrics_df

# Example usage
if __name__ == "__main__":
    filepath = 'Updated_Varied_Cs_Sb_I_Data.csv'
    target_columns = ['structure_type', 'optical_absorption', 'particle_size']
    ## consider normalizing and applying transformation to the regression targets 
    ## since uncertainty is calculated based on the variance of the predictions
    target_types = {
        'structure_type': 'classification',      # Assuming 'structure_type' is a classification target
        'optical_absorption': 'regression',      # Assuming 'optical_absorption' is a regression target
        'particle_size': 'regression'            # Assuming 'particle_size' is a regression target
    }
    num_classes_dict = {
        'structure_type': 3
    }
    target_weights = {
        'structure_type': {'unc': 1.0, 'obj': 1.0},
        'optical_absorption': {'unc': 0.0, 'obj': 1.0},
        'particle_size': {'unc': 0.0, 'obj': 1.0},
    }
    column_mapping = {
        'Ligand Type': 'ligand_type',
        'Ligand Quantity': 'ligand_quantity',
        'Additive Type': 'additive_type',
        'Additive Quantity': 'additive_quantity',
        'Halogen Type Alloy': 'halogen_type_alloy',
        'Halogen Alloy Quantity': 'halogen_alloy_quantity',
        'Temperature': 'temperature',
        'Structural Response': 'structure_type',
        'Absorption': 'optical_absorption',
        'Particle Size': 'particle_size',
    }
    categorical_cols = [ 'additive_type', 'halogen_type_alloy']
    missing_value_strategy = 'impute'
    imputation_values = None
    rows_to_remove = [0, 2, 3, 12,13,14,15, 20,21, 30,31,32,33] # hotinjection point gone.
    columns_to_remove = ['Notes', 'Sample']
    regex_columns_to_remove = ['^Unnamed']
    user_num_grid_points = { 
        'temperature': [20], # restrain to single point
        'ligand_quantity': 'unique',
        'additive_quantity': 'unique',
        'halogen_alloy_quantity': 'unique',
    }
    variable_constraints = [
    # {
    #     'conditions': {'ligand_type': '0'},
    #     'assignments': {'ligand_quantity': 0},
    #     'mutual_constraint': True, # meaning that ligand_quantity will only 
    #                                # be 0 if ligand_type is 0
    # },
    {
        'conditions': {'additive_type': '0'},
        'assignments': {'additive_quantity': 0},
        'mutual_constraint': True,
    },
    {
        'conditions': {'halogen_type_alloy': '0'},
        'assignments': {'halogen_alloy_quantity': 0},
        'mutual_constraint': True,
    }
    ]

    # Define the objective function as a string
    obj_fn_str = 'structure_type_class_2' # + 0.01 * optical_absorption - 0.01 * particle_size'
    
    # Load and preprocess data
    X, y_dict = load_and_preprocess_data(
        filepath,
        target_columns,
        target_types,
        column_mapping=column_mapping,
        categorical_cols=categorical_cols,
        missing_value_strategy=missing_value_strategy,
        imputation_values=imputation_values,
        rows_to_remove=rows_to_remove,
        columns_to_remove=columns_to_remove,
        regex_columns_to_remove=regex_columns_to_remove
    )

    # Start active learning loop
    active_learning_loop(
        X,
        y_dict,
        target_types,
        obj_fn_str,
        categorical_cols=categorical_cols,
        num_classes_dict=num_classes_dict,
        target_weights=target_weights,
        initial_train_size=None,  # Default to total size
        iterations=6,
        batch_size=8,  # Default to 8
        alpha=0.2,
        beta=0.4,
        gamma=0.4,
        num_grid_points_per_num_var=10,  # Default number of grid points for numerical variables
        user_num_grid_points=user_num_grid_points,  # Optional dict to override grid points per numerical variable
        variable_constraints=variable_constraints
    )
