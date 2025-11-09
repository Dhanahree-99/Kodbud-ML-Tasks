import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("D:/Work/ML Tasks/student_performance.csv")
print("Sample records from Dataset:\n")
print(df.head())
print("\nColumns:", df.columns.tolist())

df = df.fillna(df.mean(numeric_only=True))

for col in df.select_dtypes(include=['object']):
    df[col] = df[col].fillna(df[col].mode()[0])

label = LabelEncoder()
df['grade'] = label.fit_transform(df['grade'])

plt.figure(figsize=(10,6))
plt.scatter(df['study_hours'], df['grade'], color='blue')
plt.title("Study Hours vs Final Grade")
plt.xlabel("Study Hours per Week")
plt.ylabel("Grade (Encoded)")
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(df['attendance_percentage'], df['grade'], color='green')
plt.title("Attendance vs Final Grade")
plt.xlabel("Attendance (%)")
plt.ylabel("Grade (Encoded)")
plt.show()

features = ['study_hours', 'attendance_percentage', 'class_participation', 'total_score']
X = df[features]
y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nPredicted Final Grade (Encoded):\n",y_pred)

print("\n* Model Evaluation : *")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\n* Factor Influence: *\n", importance)
