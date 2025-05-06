import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Excel data
df = pd.read_excel(r'C:\Users\Sejal\OneDrive\Desktop\DWM prac\Linear_Regression.xlsx')

# Step 2: Extract X and Y values (assuming 1st and 2nd columns)
X = df.iloc[:, 0].tolist()
Y = df.iloc[:, 1].tolist()

# Step 3: Calculate Avg(X) and Avg(Y)
x_bar = sum(X) / len(X)
y_bar = sum(Y) / len(Y)
print(f"x_bar (Average of X) = {x_bar:.2f}")
print(f"y_bar (Average of Y) = {y_bar:.2f}")

# Step 4: Calculate β (slope)
numerator = sum((x - x_bar) * (y - y_bar) for x, y in zip(X, Y))
denominator = sum((x - x_bar) ** 2 for x in X)
beta = numerator / denominator

# Step 5: Calculate α (intercept)
alpha = y_bar - beta * x_bar

# Step 6: Display equation
print(f"\nβ (Slope) = {beta:.2f}")
print(f"α (Intercept) = {alpha:.2f}")
print(f"Linear Regression Equation: y = {alpha:.2f} + {beta:.2f}x")

# Step 7: Plot the data and regression line
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', label='Data Points')

# Step 8: Predict Y for a given X
try:
    input_x = float(input("Enter an X value to predict Y: "))
    predicted_y = alpha + beta * input_x
    print(f"Predicted Y for X = {input_x:.2f} is: {predicted_y:.2f}")
except ValueError:
    print("Invalid input for X.")

# Regression line points (handles float X too)
x_min, x_max = min(X), max(X)
x_line = [x_min + i * (x_max - x_min) / 100 for i in range(101)]
y_line = [alpha + beta * x for x in x_line]

plt.plot(x_line, y_line, color='red', label='Regression Line')

# Styling the plot
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
