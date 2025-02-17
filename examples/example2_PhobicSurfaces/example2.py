import sys
import os
# Add the project root to the path so we can import the package
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from active_sampler import active_sampling, load_and_preprocess_data


filepath = 'input.csv'

target_columns = ['contact_angle']
## consider normalizing and applying transformation to the regression targets 
## since uncertainty is calculated based on the variance of the predictions
target_types = {
   'contact_angle': 'regression',  
}

categorical_cols = ['metal_precursor', 'surface_coating_material']
# Define the objective function as a string
obj_fn_str = 'contact_angle' 

# Load and preprocess data
X, y_dict = load_and_preprocess_data(
   filepath,
   target_columns,
   target_types,
)

# Start active learning loop
active_sampling(
   X,
   y_dict,
   target_types,
   obj_fn_str,
   categorical_cols=categorical_cols,
   num_sampling=12, 
   alpha=0.25, # fn obj
   beta=0.25, # uncertainty
   gamma=0.5, # variability
   sufix = 'PhobicSurfaces',
)
