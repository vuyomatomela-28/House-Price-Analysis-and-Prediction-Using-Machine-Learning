# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make Predictions

predictions = model.predict(X_test)

predictions[:10]

#output: these are log-transformed sale prices

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

# Make Predictions

dt_predictions = dt_model.predict(X_test)

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

# Create Random Forest model

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train model

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)