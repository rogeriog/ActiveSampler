from active_sampler import active_sampling, load_and_preprocess_data

filepath = 'input.csv'

target_columns = ['structural_response']

target_types = {
   'structural_response': 'classification',      # Assuming 'structure_type' is a classification target
}
num_classes_dict = {
   'structural_response': 3
}


# Define the objective function as a string
## will use the class probability for class 2 as the objective function 
obj_fn_str = 'structural_response_class_2' 

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
   num_classes_dict=num_classes_dict,
   num_sampling=25, 
   alpha=0.25, # fn obj
   beta=0.25, # uncertainty
   gamma=0.5, # variability
   sufix = 'LARP',
)