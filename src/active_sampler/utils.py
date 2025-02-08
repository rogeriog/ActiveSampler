
import logging
from active_sampler import logger

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

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

   Parameters
   ----------
   filepath : str
       Path to the CSV file.
   target_columns : list
       List of target column names.
   target_types : dict
       Dictionary mapping target column names to their types ('classification' or 'regression').
   column_mapping : dict, optional
       Dictionary for renaming columns.
   categorical_cols : list, optional
       List of categorical column names.
   missing_value_strategy : str, optional
       Strategy to handle missing values. Options: 'drop', 'impute'.
   imputation_values : dict, optional
       Dictionary specifying the imputation values for columns.
   rows_to_remove : list, optional
       List of indices of rows to remove.
   columns_to_remove : list, optional
       List of columns to remove.
   regex_columns_to_remove : list, optional
       Regular expressions to match columns to remove.

   Returns
   -------
   X : pd.DataFrame
       Features.
   y_dict : dict
       Dictionary of target variables.
   """
   df = pd.read_csv(filepath)
   df.columns = df.columns.str.strip()
   # Trim spaces from all string values in the DataFrame
   df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
   
   # Set options to see full columns output
   pd.set_option('display.max_columns', None)

   logger.info("Initial Data (head):")
   logger.info(df.head())

   # Rename columns
   if column_mapping:
      df = df.rename(columns=column_mapping)
      logger.info("\nData after renaming columns:")
      logger.info(df.head())

   # Handle missing values
   if df.isna().any().any():  # Check if there are any missing values
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
      logger.info(df)

   # Remove duplicates
   duplicates = df[df.duplicated()]
   if not duplicates.empty:
       logger.info("\nDuplicates found:")
       logger.info(duplicates)
       df = df.drop_duplicates()
       logger.info("\nData after removing duplicates:")
       logger.info(df)

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
         logger.info(df)
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
   logger.info(X)
   logger.info("\nFinal Targets (y_dict):")
   for target, values in y_dict.items():
      logger.info(f"{target}:")
      logger.info(values)

   return X, y_dict


def get_unique_kfold_splits(X, n_splits=5, n_repeats=3, y_cat=None):
   """
   Generates unique KFold splits across multiple repeats.

   Parameters
   ----------
   X : pd.DataFrame
       Feature data.
   n_splits : int
       Number of splits.
   n_repeats : int
       Number of repeats.
   y_cat : pd.Series, optional
       Target variable for StratifiedKFold if classification.

   Returns
   -------
   splits : list
       List of (train_index, test_index) tuples.
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
