import pandas as pd
import numpy as np
import os
import gzip
import json
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix

# Paso 1: Limpieza de datos
def clear_data(df):
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.drop(columns=['ID'])
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    return df

def load_and_clean_data():
    train_data = pd.read_csv("files/input/train_data.csv.zip", index_col=False, compression="zip")
    test_data = pd.read_csv("files/input/test_data.csv.zip", index_col=False, compression="zip")

    train_data = clear_data(train_data)
    test_data = clear_data(test_data)

    return train_data, test_data

# Paso 2: División de datasets
def split_data(train_data, test_data):
    x_train = train_data.drop(columns="default")
    y_train = train_data["default"]
    x_test = test_data.drop(columns="default")
    y_test = test_data["default"]
    return x_train, y_train, x_test, y_test

# Paso 3: Creación del pipeline
def create_pipeline():
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipeline

# Paso 4: Optimización de hiperparámetros
def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [10],
        'classifier__min_samples_leaf': [4],
        "classifier__max_features": [23]
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True
    )

    model.fit(x_train, y_train)
    return model

# Paso 5: Guardar el modelo
def save_model(model):
    models_dir = 'files/models'
    os.makedirs(models_dir, exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)

# Paso 6: Cálculo de métricas
def calculate_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
    }

    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

# Paso 7: Matrices de confusión
def calculate_confusion_matrices(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    def format_confusion_matrix(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metrics = [
        format_confusion_matrix(cm_train, 'train'),
        format_confusion_matrix(cm_test, 'test')
    ]

    output_path = 'files/output/metrics.json'
    with open(output_path, 'a') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

# Función principal para ejecutar todo
def main():
    train_data, test_data = load_and_clean_data()
    x_train, y_train, x_test, y_test = split_data(train_data, test_data)

    pipeline = create_pipeline()
    model = optimize_hyperparameters(pipeline, x_train, y_train)

    save_model(model)
    calculate_metrics(model, x_train, x_test, y_train, y_test)
    calculate_confusion_matrices(model, x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()
