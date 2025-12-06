import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv(r"C:/Users/ABID/Downloads/machine learning model/dataset.csv")

X = data[['Attendance', 'Assignments', 'Midterm']]
y = data[['FinalGradeMarks']]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
attendance = float(input("Enter Attendance (%): "))
assignments = float(input("Enter Assignments marks: "))
midterm = float(input("Enter Midterm marks: "))

print(f"Prediction for attendance={attendance}%, assignments={assignments}, midterm={midterm}:")
prediction = model.predict([[attendance, assignments, midterm]])
print(prediction)
