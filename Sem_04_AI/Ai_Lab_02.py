import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

##Q-1
def least_square_method(x_axis, y_axis):
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    x_mean = np.mean(x_axis)
    y_mean = np.mean(y_axis)
    numerator = np.sum((x_axis - x_mean) * (y_axis - y_mean))
    denominator = np.sum((x_axis - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - (slope * x_mean)
    return slope, intercept

def main():
    x_axis = [16, 12, 18, 4, 3, 10, 5, 12]
    y_axis = [87, 88, 89, 68, 78, 80, 75, 83]
    slope, intercept = least_square_method(x_axis, y_axis)
    print("Slope:", slope)
    print("Intercept:", intercept)

    y_pred = intercept + slope * np.array(x_axis)
    performance_rating = intercept + slope * 20

    print(f"Estimated Performance Rating for a faculty with 20 years of experience: {performance_rating:.2f}")
    plt.scatter(x_axis, y_axis, label='Original Data')
    plt.plot(x_axis, y_pred, color='red', label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Least Squares Method')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


##Q-2
data = pd.read_csv("G:\Course notes\BostonHousing.csv")
df = pd.DataFrame(data)
x = data['crim']
y = data['rad']
corelation = x.corr(y)
print(corelation)


X = df.drop('medv', axis=1) 
Y = df['medv']
print("Shape of input features (X):", X.shape)
print("Shape of target variable (Y):", Y.shape)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print("Training set shape:", df_train.shape)
print("Testing set shape:", df_test.shape)

model = LinearRegression()
model.fit( X,Y)
Y_pred = model.predict(X)
print("predicted houses price:",Y_pred)

mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y,Y_pred)
r2 = r2_score(Y,Y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

test_sizes = [0.4, 0.5, 0.3, 0.2, 0.1, 0.05]
errors = {}
for test_size in test_sizes:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    errors[test_size] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2 Score': r2}
for test_size, error_metrics in errors.items():
    print(f"Test size: {test_size}")
    for metric, value in error_metrics.items():
        print(f"  {metric}: {value}")
    print()

random_seeds = [2, 10, 202, 755]
for random_seed in random_seeds:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed, shuffle=True)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    print(f"\nRandom Seed: {random_seed}")
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2) Score:", r2)

