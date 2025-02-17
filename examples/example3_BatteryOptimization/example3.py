
from active_sampler import active_sampling, load_and_preprocess_data

filepath = 'input.csv'

# Define the columns in the dataset that are target variables to be predicted/optimized
target_columns = ['specific_capacity', 'capacity_retention', 'coulombic_efficiency', 'heat_generation']

# Indicate whether each target is a regression or classification problem
target_types = {
   'specific_capacity': 'regression',
   'capacity_retention': 'regression',
   'coulombic_efficiency': 'regression',
   'heat_generation': 'regression'
}

# List of columns in the dataset that contain categorical data
categorical_cols = ['cathode_material', 'electrolyte_type', 'coating_material']

# Load and preprocess the data from the specified file
# This function typically handles data cleaning, encoding, and normalization
X, y_dict = load_and_preprocess_data(
   filepath,
   target_columns,
   target_types,
)

# Define the objective function and uncertainty combination formula
# These formulas guide the active learning process in selecting informative samples

# Objective function: Combines normalized target predictions to prioritize sampling
# The coefficients (0.4, 0.3, etc.) represent the relative importance of each target
obj_fn_str = "0.4 * norm_specific_capacity + 0.3 * norm_capacity_retention + 0.3 * norm_coulombic_efficiency - 0.6 * heat_generation"

# Uncertainty combination: Specifies how uncertainties from different targets are combined
# Each term represents the weighted uncertainty from a target variable
unc_fn_str = "0.4 * norm_specific_capacity_unc + 0.3 * norm_capacity_retention_unc + 0.2 * norm_coulombic_efficiency_unc + 0.1 * norm_heat_generation_unc"

# Execute the active sampling process with the defined parameters
active_sampling(
   X,
   y_dict,
   target_types,
   obj_fn_str=obj_fn_str,
   sufix='BatteryOptimization',
   categorical_cols=categorical_cols,
   num_sampling=15,
   alpha=0.4,
   beta=0.2,
   gamma=0.4,
   unc_fn_str=unc_fn_str,
)
