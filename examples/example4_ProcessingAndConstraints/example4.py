import sys
import os
# Add the project root to the path so we can import the package
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from active_sampler import active_sampling, load_and_preprocess_data

filepath = 'input.csv'
target_columns = ['structure_type', 'optical_absorption', 'particle_size']

target_types = {
   'structure_type': 'classification',      # Assuming 'structure_type' is a classification target
   'optical_absorption': 'regression',      # Assuming 'optical_absorption' is a regression target
   'particle_size': 'regression'            # Assuming 'particle_size' is a regression target
}
num_classes_dict = {
   'structure_type': 3
}

categorical_cols = [ 'additive_type', 'halogen_type_alloy']
missing_value_strategy = 'impute'
imputation_values = None
rows_to_remove = [0, 2, 3, 12,13,14,15, 21, 26,27,28,29, 30,31,32,33] # hotinjection point gone.
columns_to_remove = ['Notes', 'Sample']
regex_columns_to_remove = ['^Unnamed']
user_num_grid_points = { 
   'ligand_quantity': [0, 5, 10, 50, 100, 200, 300, 400, 500],
   'additive_quantity': 'unique',
   'halogen_alloy_quantity': 'unique',
}
variable_constraints = [
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
obj_fn_str = 'structure_type_class_2 + norm_optical_absorption + norm_particle_size' 

# Load and preprocess data
X, y_dict = load_and_preprocess_data(
   filepath,
   target_columns,
   target_types,
   categorical_cols=categorical_cols,
   missing_value_strategy=missing_value_strategy,
   imputation_values=imputation_values,
   rows_to_remove=rows_to_remove,
   columns_to_remove=columns_to_remove,
   regex_columns_to_remove=regex_columns_to_remove
)

# Start active learning loop
active_sampling(
   X,
   y_dict,
   target_types,
   obj_fn_str,
   categorical_cols=categorical_cols,
   num_classes_dict=num_classes_dict,
   initial_train_size=None,  # Default to total size
   num_sampling=8,  # Default to 8
   alpha=0.25,
   beta=0.25,
   gamma=0.5,
   user_num_grid_points=user_num_grid_points,  # Optional dict to override grid points per numerical variable
   variable_constraints=variable_constraints,
   sufix='LARP_advanced_features'
)
