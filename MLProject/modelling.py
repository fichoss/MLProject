import argparse
import os
import time
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="telco_preprocessing")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def load_data(input_path, train_file, test_file):
    train_path = os.path.join(input_path, train_file)
    test_path = os.path.join(input_path, test_file)
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train/test files not found in {input_path}. Expected {train_file}/{test_file}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def main():
    args = parse_args()

    # MLflow experiment
    mlflow.set_experiment("Telco_Churn_CI")
    # Basic requirement: use autolog in modelling.py
    mlflow.sklearn.autolog()

    train_df, test_df = load_data(args.input_path, args.train_file, args.test_file)

    # assume last column is target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # train
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=(args.max_depth if args.max_depth > 0 else None),
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        # Manual log metric (autolog will also log, but redundant logging is fine)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_score", float(f1))
        if roc is not None:
            mlflow.log_metric("roc_auc", float(roc))

        # ensure output dir exists
        os.makedirs(args.output_dir, exist_ok=True)

        # save model to outputs (so CI can upload easily)
        local_model_path = os.path.join(args.output_dir, f"model_{int(time.time())}.pkl")
        joblib.dump(model, local_model_path)

        # log the model artifact to mlflow run artifacts
        mlflow.log_artifact(local_model_path, artifact_path="model_files")

        print("Training done. Metrics:")
        print(f"  accuracy={acc:.4f}, f1={f1:.4f}, roc_auc={roc}")

if __name__ == "__main__":
    main()
