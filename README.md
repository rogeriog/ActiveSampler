# ActiveSampler: An Active Learning Package for Experimental Design in Chemistry and Materials Science

ActiveSampler is a Python package designed to facilitate active learning workflows specifically tailored for experimental design in chemistry and materials science. By intelligently selecting the most informative data points for labeling, ActiveSampler aims to optimize experiments, reduce costs, and accelerate discovery in these fields.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Implementation Details](#implementation-details)
   - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
   - [Sampling Grid Generation and Constraints](#sampling-grid-generation-and-constraints)
   - [Model Training and Prediction Collection](#model-training-and-prediction-collection)
   - [Uncertainty Estimation](#uncertainty-estimation)
   - [Objective Function Computation](#objective-function-computation)
   - [Acquisition Function Computation](#acquisition-function-computation)
   - [Active Learning Loop](#active-learning-loop)
4. [Strengths](#strengths)
5. [Use Cases in Chemistry and Materials Science](#use-cases-in-chemistry-and-materials-science)
6. [Conclusion](#conclusion)

---

## Introduction

In experimental sciences like chemistry and materials science, conducting experiments can be time-consuming and costly. ActiveSampler addresses this challenge by implementing an active learning framework that strategically selects the most informative experiments to perform next. By combining machine learning models with statistical methods, it helps researchers focus on experiments that are likely to yield the most valuable information, thereby optimizing resource utilization and accelerating innovation.

## Key Features

- **Customizable Data Handling**: Robust functions for loading, preprocessing, and encoding experimental data.
- **Adaptive Sampling Grid**: Generates sampling grids with user-defined constraints and variable-specific granularity.
- **Multiple Model Integration**: Supports various machine learning models, including Logistic Regression, Random Forest, and XGBoost.
- **Statistical Uncertainty Estimation**: Implements methods to estimate model uncertainty, crucial for experimental decision-making.
- **Flexible Objective Functions**: Allows for user-defined objectives to guide the selection of experiments.
- **Acquisition Function Computation**: Combines objective scores, uncertainty, and entropy gain to prioritize experiments.
- **Iterative Active Learning Loop**: Continuously updates models and selects new experiments based on the latest data.
- **Performance Tracking and Visualization**: Monitors model performance over iterations with visual outputs.

## Implementation Details

### Data Loading and Preprocessing

The `load_and_preprocess_data` function is the foundation for preparing experimental data:

- **Data Importing**: Reads data from CSV files into Pandas DataFrames, ensuring compatibility with various data formats common in experimental datasets.
- **Column Renaming**: Users can map raw data columns to meaningful feature names, improving code readability and maintainability.
- **Missing Value Handling**: Offers strategies like dropping incomplete rows or imputing missing values based on context, which is critical in experimental datasets where missing data is common.
- **Data Cleaning**: Removes duplicates and irrelevant columns, ensuring the dataset is clean and suitable for modeling.
- **Categorical Encoding**: One-hot encodes categorical variables, converting them into a format that machine learning models can interpret.
- **Feature-Target Splitting**: Separates the DataFrame into features (`X`) and targets (`y_dict`), accommodating multiple target variables for complex experimental outcomes.

### Sampling Grid Generation and Constraints

ActiveSampler generates a sampling grid that represents potential experiments:

- **Grid Generation**: The `generate_sampling_grid` function creates a Cartesian product of feature values, considering both numerical ranges and categorical options.
- **Variable-Specific Grid Points**: Users can specify the number of grid points for each variable, allowing finer control over important experimental parameters.
- **Constraint Application**: The `apply_constraints_to_grid` function enables the incorporation of domain knowledge by applying logical constraints to the grid. For example, ensuring that certain chemical additives are only used at specific concentrations.
- **Grid Reduction**: By applying constraints, the grid size is reduced to only feasible experiments, improving computational efficiency and experimental relevance.

### Model Training and Prediction Collection

The package trains machine learning models to predict experimental outcomes:

- **Model Training**: The `train_models_and_collect_predictions` function trains multiple models for each target variable using K-Fold cross-validation, which enhances the robustness of the predictions.
- **Model Types**: Supports classification and regression models, accommodating a wide range of experimental measurements (e.g., categorical outcomes like material phases or continuous properties like absorption spectra).
- **Prediction Collection**: Collects predictions from each model and fold, which are later used for uncertainty estimation and performance evaluation.
- **Model Selection**: Evaluates models based on performance metrics (e.g., F1 score for classification, RMSE for regression) and retains the best-performing model type for each target variable.

### Uncertainty Estimation

Estimating uncertainty is crucial for identifying which experiments will provide the most information:

- **Classification Uncertainty**: Uses entropy of the predicted probability distributions to quantify uncertainty. Higher entropy indicates that the model is less confident about the prediction.
  - **Entropy Calculation**: For a sample with predicted class probabilities \( p_1, p_2, \ldots, p_K \), entropy \( H \) is computed as:
    \[
    H = -\sum_{i=1}^{K} p_i \log(p_i)
    \]
- **Regression Uncertainty**: Calculates the variance of predictions across different models, with higher variance indicating greater uncertainty.
  - **Variance Calculation**: For predictions \( y_1, y_2, \ldots, y_N \) from \( N \) models, variance \( \sigma^2 \) is:
    \[
    \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2
    \]
    where \( \bar{y} \) is the mean prediction.

### Objective Function Computation

ActiveSampler allows users to define custom objective functions that reflect their experimental goals:

- **Expression Parsing**: Users provide an objective function as a string expression involving target variables, such as `'structure_type_class_2 + 0.01 * optical_absorption - 0.01 * particle_size'`.
- **Variable Mapping**: The `compute_objective` function maps variables in the expression to the corresponding predictions, handling both classification probabilities and regression outputs.
- **Efficient Evaluation**: Utilizes `numexpr` for fast computation over large datasets, which is essential when dealing with extensive sampling grids.

### Acquisition Function Computation

The acquisition function guides the selection of the next experiments to perform:

- **Components**:
  - **Objective Score (\( O \))**: Represents how well a sample aligns with the desired experimental outcomes.
  - **Uncertainty (\( U \))**: Reflects the model's confidence in its predictions for a sample.
  - **Entropy Gain (\( E \))**: Measures the potential increase in knowledge by selecting a sample, based on its similarity to existing data.
- **Combination**: The acquisition function \( A \) is computed as:
  \[
  A = \alpha \times O + \beta \times U + \gamma \times E
  \]
  where \( \alpha, \beta, \gamma \) are user-defined weights that balance the importance of each component.
- **Normalization**: Each component is normalized to the [0, 1] range to ensure they contribute proportionally.

### Active Learning Loop

The core of ActiveSampler is the iterative active learning process:

1. **Initialization**: Starts with an initial training set (possibly from existing experiments) and a pool of potential experiments from the sampling grid.
2. **Model Training**: Trains models using the current labeled data to predict experimental outcomes.
3. **Uncertainty and Objective Evaluation**: Computes uncertainties and objective scores for the unlabeled pool.
4. **Entropy Gain Calculation**: Assesses the entropy gain for each sample, favoring those that are different from already explored experiments.
5. **Sample Selection**: Uses the acquisition function to select a batch of samples (experiments) with the highest scores.
6. **Labeling Simulation**: Simulates the experimental results for the selected samples. In practice, this step would involve performing the experiments and obtaining real measurements.
7. **Data Update**: Adds the newly labeled data to the training set and removes them from the pool.
8. **Iteration**: Repeats the process for a specified number of iterations or until the model performance converges.
9. **Performance Tracking**: Records performance metrics, such as F1 score for classification targets or RMSE for regression targets, over iterations.
10. **Visualization**: Provides plots to visualize how model performance evolves, aiding in understanding and decision-making.

## Strengths

- **Domain-Specific Design**: Tailored for experimental design in chemistry and materials science, incorporating features like custom constraints and variable-specific sampling.
- **Efficiency**: Reduces the number of experiments needed by focusing on the most informative samples, saving time and resources.
- **Flexibility**: Supports multiple targets and allows for custom objective functions, accommodating complex experimental goals.
- **Model Robustness**: Utilizes ensemble methods and cross-validation to improve prediction reliability.
- **Uncertainty-Driven Selection**: Incorporates statistical methods for uncertainty estimation, enhancing the effectiveness of the active learning strategy.
- **Constraint Handling**: Allows the integration of domain knowledge through constraints, ensuring that suggested experiments are feasible and relevant.
- **Visualization Tools**: Offers performance tracking and visualization, which are crucial for evaluating progress and making informed decisions.

## Use Cases in Chemistry and Materials Science

- **Materials Discovery**: Accelerating the discovery of new materials with desired properties by focusing on compositions and processing conditions that are most promising.
- **Chemical Synthesis Optimization**: Identifying optimal reaction conditions (e.g., temperature, catalysts, solvent types) that maximize yield or selectivity.
- **Nanomaterial Design**: Guiding the synthesis of nanoparticles with specific sizes and shapes by selecting experimental parameters that influence these attributes.
- **Catalyst Development**: Optimizing catalyst formulations by exploring combinations of metals and supports that yield the highest activity.
- **Battery Materials Research**: Investigating electrode materials with better performance by efficiently sampling the compositional space.

## Conclusion

ActiveSampler provides a comprehensive and flexible framework for active learning in experimental sciences. By intelligently selecting experiments based on a combination of objectives, uncertainties, and entropy gain, it helps researchers in chemistry and materials science to optimize their experimental efforts. The package's integration of machine learning models, statistical methods, and domain-specific constraints makes it a powerful tool for accelerating discovery and innovation.

---

**Note**: Users are encouraged to adapt the components of ActiveSampler to their specific experimental setups. The package is designed to be extensible, allowing for the incorporation of additional models, custom uncertainty estimation methods, and tailored acquisition functions.