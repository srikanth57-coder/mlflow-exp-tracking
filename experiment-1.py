import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
import pickle

# Load the data
mlflow.set_experiment("experiment-2")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
data = pd.read_csv(r"C:\Users\M.Srikanth Reddy\Downloads\mushrooms.csv")
df = pd.DataFrame(data)


# Label Encoding for categorical columns
label_encoder = LabelEncoder()
if df.select_dtypes(include=['object']).shape[1] > 0:
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

df.info()

# Splitting the features and target
x = df.drop(columns='class', axis=1)
y = df['class']

# Standard scaling
scaler = StandardScaler()
X = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Logistic Regression Model
with mlflow.start_run():
    model1_svc = SVC(C=10, kernel='poly',degree =2)
    model1_svc.fit(x_train, y_train)

    # Save the model with pickle
    pickle.dump(model1_svc, open("model.pkl", "wb"))

    # Make predictions
    y_pred = model1_svc.predict(x_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)

    # Log metrics with mlflow
    mlflow.log_metric("acc", acc)

    # Log model parameters
    mlflow.log_param("C", model1_svc.C)
    mlflow.log_param("kernel", model1_svc.kernel)
    mlflow.log_param("degree",model1_svc.degree)

    # Print the metrics
    print("Accuracy:", acc)

    # Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)

    # Heatmap visualization
    sb.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['class'].unique(), yticklabels=df['class'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(model1_svc,"SVC")

    mlflow.log_artifact(__file__)