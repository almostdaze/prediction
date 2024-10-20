import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle

# Load the dataset
df = pd.read_csv('salary.csv')

# Preprocessing: Convert categorical variables to numeric
df['job_type'] = df['job_type'].astype('category').cat.codes
df['education'] = df['education'].astype('category').cat.codes

# Features and target variable
X = df[['job_type', 'age', 'experience', 'education']]
y = df['salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate both models
rf_predictions = rf_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

print("Random Forest MSE:", mean_absolute_error(y_test, rf_predictions))
print("Linear Regression MSE:", mean_absolute_error(y_test, lr_predictions))

# Choose the Random Forest model (you can choose any model) and save it
with open('salary_prediction_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# If you want to use the linear regression model instead, save lr_model similarly:
# with open('salary_prediction_model.pkl', 'wb') as f:
#     pickle.dump(lr_model, f)
