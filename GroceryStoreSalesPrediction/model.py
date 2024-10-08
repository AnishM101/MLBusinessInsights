import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('GroceryStoreSalesPrediction/data.csv')

df['SalesRevenueProxy'] = df['ItemMRP'] * np.sqrt(df['ItemVisibility'])

df.drop(['ItemIdentifier', 'OutletIdentifier', 'OutletLocationType', 'OutletType'], axis = 1, inplace = True)

cat_cols = ['ItemFatContent', 'ItemType', 'OutletSize']
num_cols = ['ItemWeight', 'ItemVisibility', 'ItemMRP', 'YearsSinceEstablishment']

crit_features = ['ItemWeight', 'ItemFatContent', 'ItemVisibility', 'ItemType', 'ItemMRP', 'OutletSize', 'YearsSinceEstablishment']

X = df[crit_features]
y = df['SalesRevenueProxy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop = 'first'), cat_cols)
    ]
)

reg1 = RandomForestRegressor(random_state = 42, n_estimators = 100)
reg2 = GradientBoostingRegressor(random_state = 42, n_estimators = 100)

ensemble = VotingRegressor(
    estimators = [
        ('rfr', reg1),
        ('gbr', reg2)
    ]
)

pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ]
)

cv_scores = cross_val_score(pipeline, X, y, cv = 5, scoring = 'neg_mean_squared_error')
cv_scores = -cv_scores

print(f'Cross-Validation Mean Squared Error Scores: {cv_scores}')
print(f'Mean Cross-Validation Mean Squared Error: {cv_scores.mean()}')
print(f'Standard Deviation Of Cross-Validation Mean Squared Error: {cv_scores.std()}')

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

plt.figure(figsize = (8, 6))
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel('Actual Sales Revenue Proxy')
plt.ylabel('Predicted Sales Revenue Proxy')
plt.title('Actual vs Predicted Sales Revenue Proxy')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()

plt.figure(figsize = (8, 6))
sns.scatterplot(x = y_pred, y = y_test - y_pred)
plt.axhline(0, color = 'red', linestyle = '--', lw = 2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

print(df.isnull().sum())

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)