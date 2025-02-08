# ActiveSampler: An Active Learning Package for Experimental Design in Chemistry and Materials Science

ActiveSampler is a Python package designed to facilitate active learning workflows specifically tailored for experimental design in chemistry and materials science. By intelligently selecting the most informative data points for labeling, ActiveSampler aims to optimize experiments, reduce costs, and accelerate discovery in these fields.

## Features

- **Model Training and Prediction**: Supports both classification and regression tasks using models like Logistic Regression, Random Forest, and XGBoost.
- **Uncertainty Calculation**: Computes uncertainty for classification using entropy and for regression using variance.
- **Objective Function Evaluation**: Allows custom objective functions to guide the selection of samples.
- **Diversity and Acquisition**: Incorporates diversity measures and acquisition functions to balance exploration and exploitation.
- **Grid Sampling and Constraints**: Generates sampling grids and applies constraints to ensure valid experimental designs.
- **Active Learning Selection**: Selects the most informative samples to enhance model performance with customizable weights for objective, uncertainty, and diversity.

## Installation

ActiveSampler can be installed in two primary ways:

**1. Using pip (from PyPI - Recommended):**

   This is the easiest and recommended method. Install directly from the Python Package Index (PyPI):
```bash
   pip install active_sampler
```
This will install the package and its dependencies (NumPy, pandas, scikit-learn, XGBoost, SciPy).


**2. From Source (from the project directory):**

If you have cloned the repository or have the project directory, you can install directly from the project root:
```
git clone https://github.com/rogeriog/active_sampler.git  # Replace with your repository URL
cd active_sampler
pip install .
```
This method is useful for development or if you want to modify the source code.  It installs the package in "editable" mode, so changes to the source code are immediately reflected without needing to reinstall.

## Usage

### Example

This is an example input data to select new data points in a LARP synthesis, full data on [examples/example1_LARP/input.csv](examples/example1_LARP/input.csv):

```csv
ligand_quantity,ligand_ii_quantity,halogen_alloy_quantity,antisolvent_quantity,structural_response
10.0,300,0,3000,1
5.0,300,0,3000,1
...
```
Here is the code to sample these new points:

```python
from active_sampler import active_sampling, load_and_preprocess_data

# Define the path to your data file
filepath = 'input.csv'

# Specify target columns and their types
target_columns = ['structural_response']
target_types = {
    'structural_response': 'classification',
}
num_classes_dict = {
    'structural_response': 3
}

# Define the objective function as a string
obj_fn_str = 'structural_response_class_2'

# Load and preprocess data
X, y_dict = load_and_preprocess_data(
    filepath,
    target_columns,
    target_types,
)

# Start active learning selection
active_sampling(
    X,
    y_dict,
    target_types,
    obj_fn_str,
    num_classes_dict=num_classes_dict,
    num_sampling=25,
    alpha=0.25,  # Objective weight
    beta=0.25,  # Uncertainty weight
    gamma=0.5,  # Diversity weight
    sufix='LARP',
)
```

### Input Data Format

The input data should be in CSV format:

```csv
ligand_quantity,ligand_ii_quantity,halogen_alloy_quantity,antisolvent_quantity,structural_response
10.0,300,0,3000,1
5.0,300,0,3000,1
...
```

### `load_and_preprocess_data` Function

The `load_and_preprocess_data` function loads, cleans, and prepares your data. It handles renaming, missing values, removing rows/columns, and splitting data into features (X) and targets (y_dict).  See the examples for detailed usage.

**Parameters:** `filepath`, `target_columns`, `target_types`, `column_mapping` (optional), `categorical_cols` (optional), `missing_value_strategy` (optional), `imputation_values` (optional), `rows_to_remove` (optional), `columns_to_remove` (optional), `regex_columns_to_remove` (optional).

### `active_sampling` Function Parameters

- `X`: Feature DataFrame.
- `y_dict`: Dictionary mapping target names to their Series.
- `target_types`: Dictionary mapping target names to 'classification' or 'regression'.
- `obj_fn_str`: String defining the objective function.  References:
    - Classification: `target_class_i` (e.g., `'structure_type_class_2'`).
    - Regression: `target` (e.g., `'contact_angle'`).
    - Normalized Regression: `norm_target` (e.g., `norm_contact_angle`).
- `sufix`: Suffix for output files.
- `categorical_cols`: List of categorical columns.
- `num_classes_dict`: Dictionary mapping classification targets to number of classes.
- `initial_train_size`: Initial training set size (or `None` for all data).
- `num_sampling`: Number of samples to select.
- `alpha`, `beta`, `gamma`: Weights for objective, uncertainty, and diversity.
- `user_num_grid_points`: Custom grid points per numerical variable (int, 'unique', or dict).
- `variable_constraints`: Constraints to filter the sampling grid (list of dicts).  Each dict has `conditions`, `assignments`, and optional `mutual_constraint`.
- `unc_fn_str`: Custom formula for combining uncertainties. References: `target_unc`, `norm_target_unc`.
- `diversity_settings`: Settings for diversity: `neighbor_distance_metric` (default: 'euclidean'), `same_cluster_penalty` (default: 0.5), `number_of_clusters` (default: 'num_sampling').

### Output

The `active_sampling` function generates a `.txt` file and a `.csv` file containing the coordinates of the selected samples, sorted by all columns.  See the examples folder for detailed output formats.

### Examples

The package includes several examples demonstrating different use cases, located in the `examples` folder. The structure is as follows:

```
├── README.md
├── active_sampler
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── examples
│   ├── example1_LARP
│   │   ├── example1.py
│   │   ├── input.csv
│   │   ├── selected_samples_LARP.csv
│   │   └── selected_samples_LARP.txt
│   ├── example2_PhobicSurfaces
│   │   ├── example2.py
│   │   ├── input.csv
│   │   ├── selected_samples_PhobicSurfaces.csv
│   │   └── selected_samples_PhobicSurfaces.txt
│   ├── example3_BatteryOptimization
│   │   ├── example3.py
│   │   ├── input.csv
│   │   ├── selected_samples_BatteryOptimization.csv
│   │   └── selected_samples_BatteryOptimization.txt
│   └── example4_ProcessingAndConstraints
│       ├── example4.py
│       ├── input.csv
│       ├── selected_samples_LARP_advanced_features.csv
│       └── selected_samples_LARP_advanced_features.txt
```

Each example folder contains:

-   `example[N].py`: The Python script implementing the active learning workflow.
-   `input.csv`: The input data used for the example.
> Pre-generated output files are provided for each example:
-   `selected_samples_[sufix].csv`:  The CSV file with the selected samples.
-   `selected_samples_[sufix].txt`: The text file with the selected samples and run information.

Here's a breakdown of each example:

- **`example1_LARP`**: A basic example focused on optimizing a **LARP (Ligand-Assisted Reprecipitation)** synthesis. It uses a single classification target (`structural_response`) to predict the structural outcome of the synthesis.

- **`example2_PhobicSurfaces`**: This example deals with predicting the **contact angle** of surfaces, a regression problem. It also demonstrates the use of categorical features (`metal_precursor`, `surface_coating_material`).

- **`example3_BatteryOptimization`**: A more complex, multi-output example focused on **battery material optimization**. It involves multiple regression targets (specific capacity, capacity retention, etc.) and uses custom objective and uncertainty functions to guide the selection process.  It also uses categorical features.

- **`example4_ProcessingAndConstraints`**: This example showcases advanced features like **custom grid points** (restricting the sampling space for certain variables), **variable constraints** (ensuring logical relationships between variables), and more detailed data preprocessing options. It uses a combination of classification and regression targets.

Run them directly (e.g., `python example1_LARP/example1.py`) after ensuring the `active_sampler` package is installed and the `input.csv` files are present.

## Contributing

Contributions are welcome! Please submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please contact [rogeriog.em@gmail.com].
