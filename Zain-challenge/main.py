import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def convert_comma_to_period(value):
    if isinstance(value, str):
        return value.replace(',', '.')
    return value

train_data = pd.read_csv('TrainData.csv', sep='|')

for column in train_data.columns:
    train_data[column] = train_data[column].apply(convert_comma_to_period)

numeric_columns = train_data.columns.drop(['ID', 'Season'])
train_data[numeric_columns] = train_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

X = train_data.drop(['ID', 'RH'], axis=1)
y = train_data['RH']

mask = ~y.isna()
X = X[mask]
y = y[mask]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = ['Season']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

gb = GradientBoostingRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
lgbm = LGBMRegressor(random_state=42)

voting_regressor = VotingRegressor([
    ('gb', gb),
    ('rf', rf),
    ('xgb', xgb),
    ('lgbm', lgbm)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(GradientBoostingRegressor(n_estimators=100, random_state=42))),
    ('regressor', voting_regressor)
])

param_grid = {
    'regressor__gb__n_estimators': [100, 200],
    'regressor__gb__max_depth': [5, 7],
    'regressor__gb__learning_rate': [0.01, 0.1],
    'regressor__rf__n_estimators': [100, 200],
    'regressor__rf__max_depth': [5, 7],
    'regressor__xgb__n_estimators': [100, 200],
    'regressor__xgb__max_depth': [5, 7],
    'regressor__xgb__learning_rate': [0.01, 0.1],
    'regressor__lgbm__n_estimators': [100, 200],
    'regressor__lgbm__max_depth': [5, 7],
    'regressor__lgbm__learning_rate': [0.01, 0.1],
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best R² score:", grid_search.best_score_)

best_model = grid_search.best_estimator_

test_data = pd.read_csv('TestData.csv', sep='|')

for column in test_data.columns:
    test_data[column] = test_data[column].apply(convert_comma_to_period)

numeric_columns = test_data.columns.drop(['ID', 'Season'])
test_data[numeric_columns] = test_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

X_test = test_data.drop(['ID'], axis=1)

predictions = best_model.predict(X_test)

predictions[predictions < 0] = -200
predictions[np.isnan(predictions)] = -200

result = pd.DataFrame({
    'ID': test_data['ID'],
    'RH': predictions
})

result.to_csv('result.csv', index=False)

print("Predictions saved to result.csv")