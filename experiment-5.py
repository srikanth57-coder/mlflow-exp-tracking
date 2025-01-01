import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
import pickle
import dagshub
dagshub.init(repo_owner='srikanth57-coder', repo_name='my-first-repo', mlflow=True)


# Load the data
mlflow.set_experiment("experiment-6_combination_of_all_hp")
mlflow.set_tracking_uri("https://dagshub.com/srikanth57-coder/my-first-repo.mlflow")
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

dtc = DecisionTreeClassifier()

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}
search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)



# Logistic Regression Model
with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:
    search.fit(x_train, y_train)

    # Log all hyperparameter combinations
    for i in range(len(search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination {i+1}", nested=True) as child_run:
            params = search.cv_results_['params'][i]
            mean_test_score = search.cv_results_['mean_test_score'][i]

            # Log the parameters and their corresponding score
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", mean_test_score)

    # Best hyperparameters
    print("Best parameters found: ", search.best_params_)

    best_dtc = search.best_estimator_
    best_dtc.fit(x_train, y_train)
    

    # Save the model with pickle
    pickle.dump(best_dtc, open("model.pkl", "wb"))

    model = pickle.load(open('model.pkl',"rb"))

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)

    # Log metrics with mlflow
    mlflow.log_metric("acc", acc)

    # Log model parameters
    mlflow.log_param("best_params", search.best_params_)


    df1 = mlflow.data.from_pandas(data)


    cm= confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sb.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matric")

    plt.savefig("confusion_matrix.png")

    mlflow.log_input(df1)

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(search.best_estimator_,"best_model")

    mlflow.log_artifact(__file__)

    print("acc",acc)