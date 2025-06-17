import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import r_regression
from category_encoders import TargetEncoder


class IBSClassifier:
    def __init__(self, encoding='label', scale=True, param_grid=None, model_dict=None):
        self.encoding = encoding
        self.scale = scale
        self.param_grid = param_grid or {}
        self.model_dict = model_dict or {}
        self.encoders = {}
        self.selected_features = None
        self.scaler = None
        self.imputer = None
        self.label_encodings = {}

    def load_data(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found at: {path}")
        return pd.read_csv(path)

    def preprocess_data(self, df, columns_to_drop=[], categorical_cols=[]):
        """
        Drop unnecessary columns, 
        Encode categorical features, 
        Handle missing values
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        import numpy as np

        df = df.copy()
        encodings: dict[str, np.ndarray] = {}

        # Drop unwanted columns
        if columns_to_drop:
            drop_list = [c for c in columns_to_drop if c in df.columns]
            df = df.drop(columns=drop_list)

        # Label‐encode only the columns listed in categorical_cols
        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encodings[col] = le.classes_.copy()

        # Identify numeric columns and build the pipeline
        num_list = df.select_dtypes(include=[np.number]).columns.tolist()
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
        df[num_list] = num_pipeline.fit_transform(df[num_list])

        # Store encodings for future use if needed
        self.label_encodings = encodings
        self.imputer = num_pipeline.named_steps['imputer']

        return df, encodings 
        
    def separate_features_target(self, df, target, columns_to_remove=None):
        columns_to_remove = columns_to_remove or []
        to_drop = set(columns_to_remove + [target])
        X = df.drop(columns=[col for col in to_drop if col in df.columns])
        y = df[target]
        return X, y

    def select_features(self, X, y, threshold=0.1):
        correlations = pd.Series(r_regression(X, y), index=X.columns)
        self.selected_features = correlations[correlations.abs() >= threshold].index.tolist()
        print(f"Selected {len(self.selected_features)} out of {X.shape[1]} features.")
        return self.selected_features, correlations
    
    def run_grid_search(self, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1):
        """
        Perform GridSearchCV across all models and parameter sets.

        Returns:
            A dictionary with model names as keys and best estimators/params/scores as values.
        """
        from sklearn.model_selection import GridSearchCV
        import pandas as pd

        if not self.param_grid or not self.model_dict:
            raise ValueError("Both 'param_grid' and 'model_dict' must be provided to run grid search.")

        grid_results = {}

        for name, params in self.param_grid.items():
            print(f"Running GridSearchCV for {name}...")

            if name not in self.model_dict:
                print(f"  ⚠️ Skipping {name} — not found in model_dict.")
                continue

            model = self.model_dict[name]()
            clf = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs
            )

            clf.fit(X_train, y_train)

            grid_results[name] = {
                'best_estimator': clf.best_estimator_,
                'best_params': clf.best_params_,
                'best_score': clf.best_score_,
                'cv_results': clf.cv_results_
            }

        # Display summary
        for name, result in grid_results.items():
            print(f"\n===== {name} =====")
            print("Best params:", result['best_params'])
            print(f"Best CV score: {result['best_score']:.4f}")
            df_cv = pd.DataFrame(result['cv_results'])
            display(df_cv)

        return grid_results

    def train_and_evaluate(self, model, X_train, X_test, y_train, y_test, cmap="Blues"):
        if self.scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual: 0", "Actual: 1"], columns=["Predicted: 0", "Predicted: 1"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap=cmap)
        plt.title(f"{model.__class__.__name__} Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()

        auc = None
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
                print(f"AUC: {auc:.4f}")
            else:
                auc = roc_auc_score(y_test, y_pred)
                print(f"AUC (label-based): {auc:.4f}")
        except Exception:
            print("AUC could not be computed.")

        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

        return auc, mcc, y_test, y_pred

    def train_final_model(self, model, X, y, save_path):
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        model.fit(X, y)

        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Final model trained on all data and saved to {save_path}")
        return model

    def evaluate_model(self, model, X, y, runs=30, test_size=0.2, subset_name=None, save_path=None, cmap="Blues"):
        import copy

        metrics = {'auc': [], 'prauc': [], 'mcc': []}
        best_auc = 0.0
        best_model = None
        best_cm = None
        best_y_test = None
        best_y_pred = None

        for i in range(runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            if self.scale:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_scores = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(X_test)
            )

            auc_val = roc_auc_score(y_test, y_scores)
            prauc_val = average_precision_score(y_test, y_scores)
            mcc_val = matthews_corrcoef(y_test, y_pred)

            metrics['auc'].append(auc_val)
            metrics['prauc'].append(prauc_val)
            metrics['mcc'].append(mcc_val)

            print(f"[Run {i+1}] AUC: {auc_val:.4f}, PR AUC: {prauc_val:.4f}, MCC: {mcc_val:.4f}")

            if auc_val > best_auc:
                best_auc = auc_val
                best_model = copy.deepcopy(model)
                best_cm = confusion_matrix(y_test, y_pred)
                best_y_test = y_test
                best_y_pred = y_pred
                print(f"  ↳ New best model (AUC={best_auc:.4f})")

        results = {k: self.summarize(v) for k, v in metrics.items()}

        # Show final confusion matrix only for best model
        if best_cm is not None:
            cm_df = pd.DataFrame(best_cm, index=["Actual: 0", "Actual: 1"], columns=["Predicted: 0", "Predicted: 1"])
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap=cmap)
            plt.title(f"{subset_name} Confusion Matrix (Best Run)")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.show()

        if save_path is not None and best_model is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(best_model, save_path)
            print(f"Best model (highest AUC: {best_auc:.4f}) saved to {save_path}")

        # Plot metric distributions
        for metric, values in metrics.items():
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=values)
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.show()

        return results, best_y_test, best_y_pred, X_test

    @staticmethod
    def summarize(values):
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
