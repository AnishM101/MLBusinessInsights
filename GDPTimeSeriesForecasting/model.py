import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('GDPTimeSeriesForecasting/data.csv')

df['Year'] = df['Year-Quarter'].str[:4].astype(int)
df['Quarter'] = df['Year-Quarter'].str[-1].astype(int)
df['Lag1'] = df['GDP (Billion USD)'].shift(1)
df['Lag2'] = df['GDP (Billion USD)'].shift(2)
df['RollingMean'] = df['GDP (Billion USD)'].rolling(window = 4).mean()
df['RollingStd'] = df['GDP (Billion USD)'].rolling(window = 4).std()
df['RollingMin'] = df['GDP (Billion USD)'].rolling(window = 4).min()
df['RollingMax'] = df['GDP (Billion USD)'].rolling(window = 4).max()
df['GDPQuarterDiff'] = df['GDP (Billion USD)'].diff(1)
df['GDPYearDiff'] = df['GDP (Billion USD)'].diff(4)
df['QuarterSin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
df['QuarterCos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

df.drop('Year-Quarter', axis = 1, inplace = True)

df.dropna(inplace = True)

cat_cols = ['Quarter']
num_cols = ['Year', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd', 'RollingMin', 'RollingMax', 'GDPQuarterDiff', 'GDPYearDiff', 'QuarterSin', 'QuarterCos']

crit_features = ['Year', 'Quarter', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd', 'RollingMin', 'RollingMax', 'GDPQuarterDiff', 'GDPYearDiff', 'QuarterSin', 'QuarterCos']

X = df[crit_features]
y = df['GDP (Billion USD)']

tss = TimeSeriesSplit(n_splits = 5)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop = 'first', sparse_output = False), cat_cols)
    ]
)

ridge = Ridge(alpha = 1.0)

pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('ridge', ridge)
    ]
)

final_mse = []
final_mae = []
final_r2 = []

for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    final_mse.append(mse)
    final_mae.append(mae)
    final_r2.append(r2)

print(f'Mean Squared Error: {np.mean(final_mse)}')
print(f'Mean Absolute Error: {np.mean(final_mae)}')
print(f'R^2 Score: {np.mean(final_r2)}')

df['Year-Quarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)

plt.figure(figsize = (10, 5))
plt.plot(df['Year-Quarter'].iloc[test_index], y_test, color = 'blue', label = 'Actual GDP')
plt.plot(df['Year-Quarter'].iloc[test_index], y_pred, color = 'orange', label = 'Predicted GDP')
plt.xlabel('Year-Quarter')
plt.ylabel('GDP (Billion USD)')
plt.title('Actual vs Predicted GDP')
plt.xticks(rotation = 45)
plt.legend()
plt.subplots_adjust(bottom = 0.3)
plt.show()

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)