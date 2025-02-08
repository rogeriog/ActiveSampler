import logging
from active_sampler import logger

import numpy as np
import pandas as pd
import re
import random
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, root_mean_squared_error, mean_absolute_error, r2_score,
    f1_score, precision_score, recall_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from scipy.stats import entropy
import numexpr as ne
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# Suppress warnings from sklearn, xgboost and pandas
import warnings
from sklearn.exceptions import ConvergenceWarning  # Import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning  # Import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", module="pandas.core.arrays.masked")

from active_sampler.utils import get_unique_kfold_splits 

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

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


def calculate_uncertainty_classification(predictions_df, num_classes):
   """
   Calculates uncertainty for classification tasks using the entropy of the average predicted probabilities across an ensemble of models.

   Parameters
   ----------
   predictions_df : pd.DataFrame
       DataFrame with columns for each model's predicted probabilities for each sample.
       Each column contains a list of probabilities for all classes.
   num_classes : int
       Number of classes in the classification task.

   Returns
   -------
   uncertainties : pd.Series
       Series with uncertainty values (entropy) for each sample.
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

   Parameters
   ----------
   predictions_df : pd.DataFrame
       DataFrame with true values and predictions from models.

   Returns
   -------
   uncertainties : pd.Series
       Series with uncertainty values (variance) for each sample.
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

   Parameters
   ----------
   obj_fn_str : str
       Objective function as a string expression involving target variables.
       Example: '(structure_type_class_2 + 0.2 * optical_absorption) / (0.1 * particle_size)'
   predictions_dict : dict
       Dictionary where keys are target names and values are prediction DataFrames.
       - For classification targets: Each DataFrame contains columns for each model,
         and each cell contains a list of probabilities for all classes.
       - For regression targets: Each DataFrame contains columns for each model's prediction.
   target_types : dict
       Dictionary mapping target names to their types ('classification' or 'regression').
   num_classes_dict : dict
       Dictionary mapping classification target names to their number of classes.
       Example: {'structure_type': 3}

   Returns
   -------
   objectives : pd.Series
       Series with objective values for each sample, indexed by sample identifiers.
   """
   # Initialize an empty DataFrame to hold weighted predictions for all targets
   aggregated_predictions = pd.DataFrame(index=next(iter(predictions_dict.values())).index)

   # Dictionary to map variable names to column names in aggregated_predictions
   variable_column_mapping = {}
   
   # Dictionary to store max and min values for each regression target
   regression_min_max = {}

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
         # Store the min and max values for normalization
         regression_min_max[target] = {
                'min': avg_pred.min(),
                'max': avg_pred.max()
         }

      else:
         raise ValueError(f"Unknown target type for '{target}'. Expected 'classification' or 'regression'.")
    
   # Normalize regression targets and add them to the aggregated data
   for target, min_max in regression_min_max.items():
      col_name = f"{target}"
      norm_col_name = f"norm_{target}"
      variable_name = norm_col_name
      
      # Perform Min-Max scaling on the predictions
      normalized_values = (aggregated_predictions[col_name] - min_max['min']) / (min_max['max'] - min_max['min']) if min_max['max'] != min_max['min'] else 0.0
      aggregated_predictions[norm_col_name] = normalized_values
      variable_column_mapping[variable_name] = norm_col_name
   
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

def compute_acquisition(uncertainty, objective, diversity, alpha=1.0, beta=1.0, gamma=1.0):
   """
   Computes the acquisition function combining uncertainty, objective, and diversity.

   Parameters
   ----------
   uncertainty : pd.Series
       Uncertainty values for each sample.
   objective : pd.Series
       Objective values for each sample.
   diversity : pd.Series
       Diversity values for each sample.
   alpha : float
       Weight for objective.
   beta : float
       Weight for uncertainty.
   gamma : float
       Weight for diversity.

   Returns
   -------
   acquisition : pd.Series
       Acquisition values for each sample.
   """
   # Normalize uncertainty, objective, and diversity to [0, 1]
   uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) if uncertainty.max() != uncertainty.min() else 0.0
   objective_norm = (objective - objective.min()) / (objective.max() - objective.min()) if objective.max() != objective.min() else 0.0
   diversity_norm = (diversity - diversity.min()) / (diversity.max() - diversity.min()) if diversity.max() != diversity.min() else 0.0

   acquisition = alpha * objective_norm + beta * uncertainty_norm + gamma * diversity_norm

   return acquisition


def predict_with_models(X, models, model_type='classification'):
   """
   Makes predictions on X using the list of models.

   Parameters
   ----------
   X : pd.DataFrame
       Feature data.
   models : list
       List of trained models.
   model_type : str
       'classification' or 'regression'.

   Returns
   -------
   predictions_df : pd.DataFrame
       DataFrame with predictions.
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


def generate_sampling_grid(X, num_grid_points_per_num_var=10): 
   """
   Generates a sampling grid based on the ranges or unique values of each feature in X.

   Parameters
   ----------
   X : pd.DataFrame
       Feature data.
   num_grid_points_per_num_var : int, dict, or str, optional
       - If int, uses that many evenly spaced grid points (via linspace) for numerical variables.
       - If dict, the number of grid points per numerical variable is specified individually. 
         If the value is 'unique' for a column, then the unique values are used.
       - If str and set to 'unique', then the unique values are used for every numerical variable.

   Returns
   -------
   grid_df : pd.DataFrame
       DataFrame representing the sampling grid.
   """
   grid_dict = {}
   for col in X.columns:
      # Handle categorical variables or columns with <= 2 unique values
      if X[col].dtype == "object" or X[col].nunique() <= 2:
         grid_dict[col] = X[col].unique()
      else:
         # Handle numerical variables
         if isinstance(num_grid_points_per_num_var, dict):
               # Check if a custom value is specified for the column
               value = num_grid_points_per_num_var.get(col, "unique")
               if value == "unique":
                  grid_dict[col] = X[col].unique()
               elif isinstance(value, list):
                  # Use the provided list of values
                  grid_dict[col] = value
               else:
                  # Generate evenly spaced grid points
                  min_val = X[col].min()
                  max_val = X[col].max()
                  grid_dict[col] = np.linspace(min_val, max_val, num=value)
         elif num_grid_points_per_num_var == "unique":
               # If default is "unique"
               grid_dict[col] = X[col].unique()
         else:
               # Otherwise, generate grid points based on the given number
               min_val = X[col].min()
               max_val = X[col].max()
               grid_dict[col] = np.linspace(min_val, max_val, num=num_grid_points_per_num_var)

   # Log the grid values for each column
   for col, values in grid_dict.items():
      logger.info(f"Feature '{col}': {values}")

   # Create the cartesian product of grids for all features
   grid_df = pd.DataFrame(
      np.array(np.meshgrid(*grid_dict.values())).T.reshape(-1, len(grid_dict)),
      columns=grid_dict.keys()
   )

   # Restore original data types for categorical columns
   for col in X.columns:
      if X[col].dtype == "object":
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

   Parameters
   ----------
   grid_df : pd.DataFrame
       The sampling grid.
   variable_constraints : list of dicts
       List of constraints where each constraint is a dictionary
       with 'conditions', 'assignments', and optionally 'mutual_constraint' keys.

   Returns
   -------
   grid_df : pd.DataFrame
       The constrained sampling grid.
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



def active_sampling(
   X,
   y_dict,
   target_types,
   obj_fn_str,
   sufix='',
   categorical_cols=None,
   num_classes_dict=None,
   initial_train_size=None,
   num_sampling=8,
   alpha=0.25,
   beta=0.25,
   gamma=0.5,
   user_num_grid_points=None,
   variable_constraints=None,
   unc_fn_str=None,
   diversity_settings={'neighbor_distance_metric': 'euclidean',
                     'same_cluster_penalty': 0.5,
                     'number_of_clusters': 'num_sampling'},
):
   """
   Performs an active learning selection to iteratively select the most informative samples,
   enhancing model performance through strategic sampling. Handles both classification
   and regression targets, supporting multi-output optimization.

   Parameters:
   - X (pd.DataFrame): Feature dataframe.
   - y_dict (dict): Dictionary mapping target names to their respective Series.
   - target_types (dict): Dictionary mapping target names to their types ('classification' or 'regression').
   - obj_fn_str (str): String expression defining the objective function, e.g., '0.4 * target1 + 0.6 * target2'.
   - sufix (str, optional): Suffix for output files. Defaults to ''.
   - categorical_cols (list, optional): List of categorical column names in X. Defaults to None.
   - num_classes_dict (dict, optional): Dictionary mapping classification targets to their number of classes. Defaults to None.
   - initial_train_size (int, optional): Number of initial training samples. If None, uses the entire dataset. Defaults to None.
   - num_sampling (int, optional): Number of samples to select in each iteration. Defaults to 8.
   - alpha (float, optional): Weight for objective in acquisition function. Defaults to 0.25.
   - beta (float, optional): Weight for uncertainty in acquisition function. Defaults to 0.25.
   - gamma (float, optional): Weight for diversity in acquisition function. Defaults to 0.5.
   - user_num_grid_points (dict, optional): Custom number of grid points per numerical variable. Defaults to None.
   - variable_constraints (dict, optional): Constraints to filter the sampling grid. Defaults to None.
   - unc_fn_str (str, optional): Custom formula for combining uncertainties. Defaults to None.
   - diversity_settings (dict, optional): Settings for diversity calculation. Defaults to predefined settings.

   Returns:
   - None: The function modifies the training set in place and generates output files.

   Observations:
   - The default parameters for alpha, beta, and gamma should prioritize exploration (diversity and uncertainty minimization) 
   over exploitation (objective). You should probably increase alpha as you collect more data.
   """

   # Section: Generate Sampling Grid
   # --------------------------------
   # Determine the number of grid points per numerical variable
   if user_num_grid_points is not None:
      grid_points = user_num_grid_points
   else:
      grid_points = "unique"  # Default to using unique values for each feature
   
   # Generate the sampling grid based on the dataset
   X_grid = generate_sampling_grid(X, num_grid_points_per_num_var=grid_points)
   # Apply constraints to filter out invalid samples
   X_grid = apply_constraints_to_grid(X_grid, variable_constraints)
   
   # Section: Encode Categorical Variables
   # -------------------------------------
   if categorical_cols:
      # Identify original categorical columns present in the dataset
      original_categorical_cols = [col for col in categorical_cols if col in X.columns]
      
      # One-hot encode the categorical variables
      X = pd.get_dummies(X, columns=categorical_cols)
      logger.info("\nData after encoding categorical variables:")
      logger.info(X.head())
      
      # Apply the same encoding to the grid
      X_grid = pd.get_dummies(X_grid, columns=categorical_cols)
      # Fill missing columns with zeros
      missing_cols = set(X.columns) - set(X_grid.columns)
      for col in missing_cols:
         X_grid[col] = 0
      
      # Ensure consistent data types between grid and training data
      X_grid = X_grid[X.columns]
      X_grid = X_grid.astype(X.dtypes)
   
   # Section: Initialize Training and Pool Data
   # -----------------------------------------
   # Determine initial training set size
   if initial_train_size is None:
      initial_train_size = len(X)
   else:
      initial_train_size = initial_train_size
   
   if initial_train_size < len(X):
      # Randomly sample initial training set
      initial_train_indices = y_dict[next(iter(y_dict))].sample(n=initial_train_size, random_state=42).index.tolist()
      
      X_train = X.loc[initial_train_indices].copy()
      y_train_dict = {target: y.loc[initial_train_indices].copy() for target, y in y_dict.items()}
      
      # Remaining samples form the test set
      X_test = X.drop(index=initial_train_indices).copy()
      y_test_dict = {target: y.drop(index=initial_train_indices).copy() for target, y in y_dict.items()}
      
      # Combine grid and test set to form the pool
      X_pool = pd.concat([X_grid, X_test], ignore_index=True).drop_duplicates().reset_index(drop=True)
      # Remove overlapping samples with training set
      X_pool = X_pool[~X_pool.apply(tuple, axis=1).isin(X_train.apply(tuple, axis=1))]
      
      # Initialize pool labels
      y_pool_dict = {}
      for target in y_dict.keys():
         y_pool_test = y_test_dict[target].reset_index(drop=True)
         y_pool_grid = pd.Series([np.nan] * len(X_grid), name=target)
         y_pool_combined = pd.concat([y_pool_test, y_pool_grid], ignore_index=True)
         # Remove overlapping indices with training set
         y_pool_combined = y_pool_combined[~X_pool.apply(tuple, axis=1).isin(X_train.apply(tuple, axis=1))]
         y_pool_dict[target] = y_pool_combined
   else:
      # Use entire dataset as initial training set
      X_train = X.copy()
      y_train_dict = {target: y.copy() for target, y in y_dict.items()}
      
      # Pool consists of the grid only, excluding training samples
      X_pool = X_grid.copy()
      X_pool = X_pool[~X_pool.apply(tuple, axis=1).isin(X_train.apply(tuple, axis=1))]
      # Initialize pool labels with NaNs
      y_pool_dict = {target: pd.Series([np.nan] * len(X_pool), index=X_pool.index) for target in y_dict.keys()}
   
   # Log initial setup details
   logger.info(f"Initial training set size: {len(X_train)}")
   if initial_train_size < len(X):
      logger.info(f"Initial test set size: {len(X_test)}")
   logger.info(f"Pool size after removing overlaps with X_train: {len(X_pool)}")
   
   # Performance tracking
   performance_metrics = []
   
   # Track selected model types and retained models
   selected_model_types = {target: None for target in y_dict.keys()}
   retained_models = {target: [] for target in y_dict.keys()}
   
   # Section: Train Models for Each Target
   # ------------------------------------
   models_dict = {}
   predictions_train_dict = {}
   predictions_test_dict = {}
   splits_dict = {}
   model_performance = {}
   
   for target, y_train in y_train_dict.items():
      model_type = target_types[target]
      logger.info(f"Training models for target '{target}' ({model_type})")
      
      # Determine cross-validation splits
      if model_type == 'classification':
         splits = get_unique_kfold_splits(X_train, n_splits=5, n_repeats=3, y_cat=y_train)
      else:
         splits = get_unique_kfold_splits(X_train, n_splits=5, n_repeats=3)
      
      splits_dict[target] = splits
      
      # Train models and collect predictions
      predictions_train, predictions_test, models = train_models_and_collect_predictions(
         X_train, y_train, model_type=model_type, splits=splits
      )
      
      predictions_train_dict[target] = predictions_train
      predictions_test_dict[target] = predictions_test
      models_dict[target] = models
      
      # Calculate average metrics across folds
      avg_metrics_test = calculate_average_metrics(predictions_test, model_type=model_type)
      logger.info(f"Average test metrics for target '{target}':\n{avg_metrics_test}")
      
      # Select best model type based on performance
      if model_type == 'classification':
         best_row = avg_metrics_test.loc[avg_metrics_test['f1_score'].idxmax()]
         model_performance[target] = best_row['f1_score']
      elif model_type == 'regression':
         best_row = avg_metrics_test.loc[avg_metrics_test['rmse'].idxmin()]
         model_performance[target] = best_row['rmse']
      else:
         raise ValueError(f"Unknown model type '{model_type}' for target '{target}'.")
      
      best_model_type = best_row['model']
      selected_model_types[target] = best_model_type
      logger.info(f"Selected best model type for target '{target}': {best_model_type}")
      
      # Retain only the best-performing models
      retained_models[target] = models[best_model_type]
      logger.info(f"Retained {len(retained_models[target])} models of type '{best_model_type}' for target '{target}'")
   
   # Section: Make Predictions on Pool Data
   # -------------------------------------
   predictions_pool_dict = {}
   uncertainty_dict = {}
   
   for target, models in models_dict.items():
      model_type = target_types[target]
      logger.info(f"Predicting on pool data for target '{target}' ({model_type})")
      
      if not models:
         logger.warning(f"No models available for target '{target}'. Skipping.")
         continue
      
      # Generate predictions on the pool set
      predictions_pool = predict_with_models(X_pool, retained_models[target], model_type=model_type)
      predictions_pool_dict[target] = predictions_pool
      
      # Calculate uncertainty
      if model_type == 'classification':
         num_classes = len(predictions_pool.iloc[0, 0])
         uncertainty = calculate_uncertainty_classification(predictions_pool, num_classes)
      elif model_type == 'regression':
         uncertainty = calculate_uncertainty_regression(predictions_pool)
      else:
         raise ValueError(f"Unknown model type '{model_type}' for target '{target}'.")
      
      uncertainty_dict[target] = uncertainty
   
   # Normalize uncertainties to [0, 1] range
   for target, uncertainty in uncertainty_dict.items():
      uncertainty_dict[target] = (
         (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) 
         if uncertainty.max() != uncertainty.min() 
         else uncertainty
      )
   
   # Combine uncertainties using custom formula or default method
   if unc_fn_str:
      # Prepare normalized variables for the uncertainty function
      uncertainties_data = {}
      regression_min_max = {}
      
      for target, unc_series in uncertainty_dict.items():
         min_val = unc_series.min()
         max_val = unc_series.max()
         normalized_values = (unc_series - min_val) / (max_val - min_val) if max_val != min_val else 0.0
         uncertainties_data[f"norm_{target}_unc"] = normalized_values
         uncertainties_data[f"{target}_unc"] = unc_series
         regression_min_max[target] = {'min': min_val, 'max': max_val}
      
      # Use custom formula to combine uncertainties
      logger.info(f"Using custom uncertainty function: {unc_fn_str}")
      try:
         total_uncertainty = ne.evaluate(unc_fn_str, local_dict=uncertainties_data)
         total_uncertainty = pd.Series(total_uncertainty, index=X_pool.index)
      except Exception as e:
         raise ValueError(f"Error in evaluating unc_fn_str: {e}")
   else:
      # Default: Sum uncertainties with equal weights
      logger.info("Using default uncertainty combination (sum with equal weights).")
      total_uncertainty = pd.Series(0, index=X_pool.index)
      for _, uncertainty in uncertainty_dict.items():
         total_uncertainty += uncertainty / len(uncertainty_dict)
   
   # Compute the objective function
   objective = compute_objective(
      obj_fn_str,
      predictions_pool_dict,
      target_types,
      num_classes_dict
   )
   
   # Section: Diversity Measure and Acquisition
   # -----------------------------------------
   logger.info("Calculating scaler to normalize X values for entropy gain function.")
   scaler = StandardScaler()
   X_total = pd.concat([X_train, X_pool])
   scaler.fit(X_total)
   del X_total  # Free memory
   
   # Normalize training and pool data
   X_pool_scaled = pd.DataFrame(scaler.transform(X_pool), index=X_pool.index, columns=X_pool.columns)
   X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
   
   # Apply diversity settings
   neighbor_distance_metric = diversity_settings.get('neighbor_distance_metric', 'euclidean')
   same_cluster_penalty = diversity_settings.get('same_cluster_penalty', 0.5)
   num_clusters = diversity_settings.get('number_of_clusters', 5)
   
   # Cluster pool samples
   if num_clusters == 'num_sampling':
      num_clusters = num_sampling
   kmeans = KMeans(n_clusters=num_clusters, random_state=42)
   cluster_labels = kmeans.fit_predict(X_pool_scaled)
   
   # Initialize sample selection
   selected_indices = []
   selected_clusters = set()
   
   # Check if pool has enough samples
   if len(X_pool) < num_sampling:
      logger.warning("Pool size is insufficient for sampling. Skipping iteration.")
      return
   
   # Select samples iteratively
   for _ in range(num_sampling):
      # Find nearest neighbors to current training set
      nbrs = NearestNeighbors(n_neighbors=5, metric=neighbor_distance_metric).fit(X_train_scaled.values)
      distances, _ = nbrs.kneighbors(X_pool_scaled.values)
      avg_distances = distances.mean(axis=1)
      
      # Create diversity measure
      diversity = pd.Series(avg_distances, index=X_pool_scaled.index)
      total_uncertainty = total_uncertainty[diversity.index]
      objective = objective[diversity.index]
      
      # Apply diversity penalty
      number_of_penalized_samples = 0
      for idx, sample_idx in enumerate(X_pool_scaled.index):
         cluster = cluster_labels[idx]
         if cluster in selected_clusters:
               diversity[sample_idx] *= same_cluster_penalty
               number_of_penalized_samples += 1
      logger.debug(f"Number of penalized samples: {number_of_penalized_samples}")
      
      # Compute acquisition scores
      acquisition = compute_acquisition(total_uncertainty, objective, diversity, alpha=alpha, beta=beta, gamma=gamma)
      
      # Select sample with highest acquisition score
      selected_idx = acquisition.idxmax()
      selected_indices.append(selected_idx)
      
      # Debugging information
      logger.debug(f"Selected sample index: {selected_idx}")
      logger.debug(f"Objective value: {objective[selected_idx]}")
      logger.debug(f"Uncertainty value: {total_uncertainty[selected_idx]}")
      logger.debug(f"Diversity value: {diversity[selected_idx]}")
      
      if logger.isEnabledFor(logging.DEBUG):
         uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) if uncertainty.max() != uncertainty.min() else pd.Series(0, index=uncertainty.index)
         objective_norm = (objective - objective.min()) / (objective.max() - objective.min()) if objective.max() != objective.min() else pd.Series(0, index=objective.index)
         diversity_norm = (diversity - diversity.min()) / (diversity.max() - diversity.min()) if diversity.max() != diversity.min() else pd.Series(0, index=diversity.index)
         logger.debug(f"Normalized Objective value: {objective_norm[selected_idx]}")
         logger.debug(f"Normalized Uncertainty value: {uncertainty_norm[selected_idx]}")
         logger.debug(f"Normalized Diversity value: {diversity_norm[selected_idx]}")
      
      # Update training and pool sets
      X_train_scaled = pd.concat([X_train_scaled, X_pool_scaled.loc[[selected_idx]]])
      X_pool_scaled = X_pool_scaled.drop(index=selected_idx)
   
   # Clean up memory
   del scaler, X_train_scaled, X_pool_scaled
   logger.debug(f"Selected samples: {selected_indices}")
   
   # Retrieve selected samples from the grid
   X_selected = X_pool.loc[selected_indices].copy()
   
   # Debugging: Display selected samples
   with pd.option_context('display.max_rows', None, 'display.max_columns', None):
      logger.debug(f"Selected samples coordinates:\n{X_selected}")
   
   # Section: Simulate Labelling
   # ---------------------------
   y_true_selected = {}
   selected_models_info = {}
   
   for target in y_dict.keys():
      models = models_dict[target]
      if not models:
         logger.warning(f"No models available for target '{target}'. Skipping labeling.")
         y_true_selected[target] = pd.Series([np.nan] * len(X_selected), index=X_selected.index)
         selected_models_info[target] = [None] * len(X_selected)
         continue
      
      # Randomly select a model for prediction
      idx_selected = np.random.choice(len(retained_models[target]))
      selected_model = retained_models[target][idx_selected]
      selected_models_info[target] = selected_model
      
      # Predict labels for selected samples
      if target_types[target] == 'classification':
         y_pred = selected_model.predict(X_selected.values)
      elif target_types[target] == 'regression':
         y_pred = selected_model.predict(X_selected.values)
      else:
         y_pred = [np.nan] * len(X_selected)
      
      y_true_selected[target] = pd.Series(y_pred, index=X_selected.index)
      
      # Log predictions and model used
      logger.debug(f"Selected models for target '{target}': {idx_selected}")
      logger.debug(f"Predicted labels for target '{target}':\n{y_true_selected[target]}")
   
   # Ensure consistent data types
   X_selected = X_selected.astype(X_train.dtypes)
   
   # Update training set and pool
   X_train = pd.concat([X_train, X_selected])
   logger.info(f"Updated training set size: {len(X_train)}")
   logger.info(f"Updated pool size: {len(X_pool) - len(X_selected)}")
   
   # Update labels in training set
   for target in y_dict.keys():
      y_train_dict[target] = pd.concat([y_train_dict[target], y_true_selected[target]])
   
   # Remove selected samples from pool
   X_pool = X_pool.drop(index=selected_indices)
   for target in y_dict.keys():
      y_pool_dict[target] = y_pool_dict[target].drop(index=selected_indices)
   
   # Section: Output Performance Metrics
   # -----------------------------------
   logger.info("\nModel Performance Summary:")
   logger.info("-" * 80)
   
   for target in y_dict.keys():
      logger.info(f"\nTarget: {target}")
      logger.info("-" * 40)
      
      selected_model = selected_models_info[target].__class__.__name__
      metric_value = model_performance[target]
      metric_type = 'F1 Score' if target_types[target] == 'classification' else 'RMSE'
      
      logger.info(f"Selected Model: {selected_model}")
      logger.info(f"{metric_type}: {metric_value:.4f}")
   
   # Section: Save Selected Samples
   # -----------------------------
   if categorical_cols:
      for category in original_categorical_cols:
         one_hot_columns = [col for col in X_selected.columns if col.startswith(f"{category}_")]
         X_selected[category] = X_selected[one_hot_columns].idxmax(axis=1).str[len(category) + 1:]
         X_selected = X_selected.drop(columns=one_hot_columns)
   
   # Sort selected samples by all columns
   sorted_samples = X_selected.sort_values(by=X_selected.columns.tolist())
   
   # Log sorted samples
   logger.info("\nSelected samples ordered by all columns sequentially:")
   logger.info("-" * 80)
   column_widths = {col: max(len(str(val)) for val in sorted_samples[col]) for col in sorted_samples.columns}
   header = "|".join(f"{col:<{column_widths[col]}}" for col in sorted_samples.columns)
   logger.info(header)
   logger.info("-" * len(header))
   for _, row in sorted_samples.iterrows():
      formatted_row = "|".join(f"{str(val):<{column_widths[col]}}" for col, val in row.items())
      logger.info(formatted_row)
    
   # Save to file
   sufix_txt = sufix if sufix == '' else f"_{sufix}"
   with open(f'selected_samples{sufix_txt}.txt', 'a') as f:
      f.write(f"\n=== Basic info on the run ===\n")
      f.write(f"Grid size: {len(X_grid)}\n")
      f.write(f"considers all possible sampling points based on user data and settings.\n\n")
      f.write(f"Initial training set size: {len(X_train) - len(X_selected)}\n")
      f.write(f"number of sampling points which target properties were measured.\n\n")
      f.write(f"Alpha (weight for objective): {alpha}\n")
      f.write(f"Beta (weight for uncertainty): {beta}\n")
      f.write(f"Gamma (weight for diversity): {gamma}\n")
      f.write(f"Alpha encourages sampling to maximize the objective function (e.g. maximizing target probability).\n")
      f.write(f"Beta encourages sampling in points of high uncertainty of the predictions.\n")
      f.write(f"Gamma encourages variability in the selected samples.\n\n")
      f.write(f"\n=== Selected Samples ===\n")
      f.write("Ordered by each column sequentially\n")
      f.write("-" * 80 + "\n")
      f.write(header + "\n")
      f.write("-" * len(header) + "\n")
      for _, row in sorted_samples.iterrows():
            formatted_row = "|".join(f"{str(val):<{column_widths[col]}}" for col, val in row.items())
            f.write(formatted_row + "\n")
      f.write("\n")
   # Save the selected samples to a CSV file
   sorted_samples.to_csv(f'selected_samples{sufix_txt}.csv', index=False)
   logger.info(f"Selected samples saved to 'selected_samples{sufix_txt}.csv' and 'selected_samples{sufix_txt}.txt'.")
   