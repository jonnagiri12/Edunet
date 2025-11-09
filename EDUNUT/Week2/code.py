import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset (filename: ENB2012_data.xlsx)
df = pd.read_excel(r'C:\learning\EDUNUT\Week2\ENB2012_data.xlsx')

# 2. Rename columns for easy use (optional)
df.columns = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
              'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution',
              'Heating_Load', 'Cooling_Load']

# 3. Choose features and target (here, we try to predict 'Heating_Load')
features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
            'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution']
target = 'Heating_Load'

X = df[features]
y = df[target]

# 4. Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict on test data
y_pred = model.predict(X_test)

# 7. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Absolute Error:', mae)
print('R2 Score:', r2)

# 8. Predict for a new building example
example = pd.DataFrame([[0.75, 514.5, 294.0, 110.25, 3.5, 2, 0.10, 1]],
            columns=['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
                     'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'])
predicted = model.predict(example)
print('Predicted Heating Load for example:', predicted[0])