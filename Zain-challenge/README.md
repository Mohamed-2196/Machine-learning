How to Open and Run the Code
1. Prerequisites
Python: Ensure Python 3.7 or higher is installed.

Libraries: Install the required libraries using pip:

bash
Copy
pip install pandas numpy scikit-learn xgboost lightgbm
Dataset: Ensure the dataset files (TrainData.csv and TestData.csv) are available in the same directory as the script.

2. Running the Code
Save the provided code in a Python file, e.g., train_and_predict.py.

Open a terminal or command prompt and navigate to the directory where the script is saved.

Run the script:

bash
Copy
python main.py
3. Output
The script will:

Load and preprocess the training and test data.

Train a Voting Regressor combining Gradient Boosting, Random Forest, XGBoost, and LightGBM models.

Perform hyperparameter tuning using GridSearchCV.

Generate predictions on the test data.

Save the predictions to a CSV file named result.csv.

4. Troubleshooting
Missing Dataset: Ensure TrainData.csv and TestData.csv are present in the working directory.

Library Errors: If any library is missing, install it using pip install <library_name>.

Data Format Issues: Ensure the data is in the correct format (e.g., numeric columns should not contain invalid characters).

5. Customization
Model Selection: Modify the VotingRegressor to include or exclude specific models.

Hyperparameter Tuning: Adjust the param_grid to explore different hyperparameter combinations.

Feature Engineering: Modify the preprocessor to include additional transformations or feature engineering steps.

6. Dependencies
The code uses the following libraries:

pandas for data manipulation.

numpy for numerical operations.

scikit-learn for machine learning pipelines, preprocessing, and model evaluation.

xgboost and lightgbm for advanced boosting algorithms.
