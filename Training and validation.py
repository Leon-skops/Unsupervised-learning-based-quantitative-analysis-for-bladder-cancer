import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import os

# --- Configuration ---
TRAINING_DATA_PATH = 'path_to_training_data_with_optimal_features.csv'
TARGET_COLUMN_NAME = 'your_target_column_name'
CLASS_LABELS = [0, 1]

# --- Data loading ---
print("--- Loading Training Data ---")
try:
    training_data = pd.read_csv(TRAINING_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Training data file not found at {TRAINING_DATA_PATH}")
    exit()

if TARGET_COLUMN_NAME not in training_data.columns:
    print(f"Error: Target column '{TARGET_COLUMN_NAME}' not found in training data.")
    exit()

X_train_full = training_data.drop(TARGET_COLUMN_NAME, axis=1)
y_train_series = training_data[TARGET_COLUMN_NAME]

if not isinstance(y_train_series, pd.Series):
    print("Warning: y_train_series was not a Pandas Series after selection. This is unexpected.")

    if isinstance(y_train_series, pd.DataFrame) and y_train_series.shape[1] == 1:
        y_train_series = y_train_series.squeeze()
    else:
        print(f"Error: y_train_series is not a 1D array-like structure. Shape: {y_train_series.shape}")
        exit()


selected_feature_names = list(X_train_full.columns)
X_train_selected_np = X_train_full.values

print(f"Training data loaded: X_train_full shape {X_train_full.shape}, y_train_series shape {y_train_series.shape}")
print(f"Number of selected features: {len(selected_feature_names)}")

# --- Model configurations ---
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear'],
            'class_weight': ['balanced', None]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=2000),
        'params': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', None]
        }
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
}

# --- Nested cross-validation (5x5-fold) ---
print("\n--- Starting Nested Cross-Validation ---")
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_train_selected_np, y_train_series), 1):
    print(f"\n=== Outer Fold {fold} ===")
    X_tr_fold, X_val_fold = X_train_selected_np[train_idx], X_train_selected_np[test_idx]

    y_tr_series_fold, y_val_series_fold = y_train_series.iloc[train_idx], y_train_series.iloc[test_idx]

    scaler_fold = StandardScaler()
    X_tr_scaled_fold = scaler_fold.fit_transform(X_tr_fold)
    X_val_scaled_fold = scaler_fold.transform(X_val_fold)

    for model_name, config in models.items():
        print(f"  Tuning {model_name}...")
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            error_score='raise'
        )
        try:
            grid.fit(X_tr_scaled_fold, y_tr_series_fold)
        except ValueError as e:
            print(f"    Error during GridSearchCV for {model_name}: {e}")
            print(f"    Skipping {model_name} for this fold.")
            outer_results.append({
                'fold': fold,
                'model': model_name,
                'auc': np.nan,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'best_params': {}
            })
            continue

        best_model_for_fold = grid.best_estimator_


        try:
            y_pred_proba = best_model_for_fold.predict_proba(X_val_scaled_fold)[:, 1]
            auc = roc_auc_score(y_val_series_fold, y_pred_proba)
        except Exception as e:
            print(f"    Could not calculate AUC for {model_name}: {e}")
            auc = np.nan
            y_pred_proba = np.array([0.5] * len(y_val_series_fold))

        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        cm = confusion_matrix(y_val_series_fold, y_pred_binary, labels=CLASS_LABELS)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        outer_results.append({
            'fold': fold,
            'model': model_name,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'best_params': grid.best_params_
        })
        print(
            f"  {model_name}: AUC={auc:.3f}, Sens={sensitivity:.3f}, Spec={specificity:.3f}, Best Params: {grid.best_params_}")

# --- Result analysis ---
print("\n--- Nested Cross-Validation Results ---")
results_df = pd.DataFrame(outer_results)
mean_metrics = results_df.groupby('model')[['auc', 'sensitivity', 'specificity']].mean()
std_metrics = results_df.groupby('model')[['auc', 'sensitivity', 'specificity']].std()

print("\n=== Average Performance Across Outer Folds (Mean) ===")
print(mean_metrics.sort_values('auc', ascending=False))
print("\n=== Average Performance Across Outer Folds (Std Dev) ===")
print(std_metrics.sort_values('auc', ascending=False))

# --- Train final model on full dataset ---
best_model_type_name = mean_metrics.dropna(subset=['auc']).sort_values('auc', ascending=False).index[0]
print(f"\n--- Training Final Model ---")
print(f"Best performing model type (average AUC from nested CV): {best_model_type_name}")

best_model_config = models[best_model_type_name]

final_scaler = StandardScaler()
X_train_selected_scaled_full = final_scaler.fit_transform(X_train_selected_np)

print(f"\nRetuning {best_model_type_name} on the full training dataset...")

final_tuning_cv = KFold(n_splits=5, shuffle=True, random_state=42)
final_grid = GridSearchCV(
    estimator=best_model_config['model'],
    param_grid=best_model_config['params'],
    cv=final_tuning_cv,
    scoring='roc_auc',
    n_jobs=-1,
    refit=True,
    error_score='raise'
)
final_grid.fit(X_train_selected_scaled_full, y_train_series)

final_best_model = final_grid.best_estimator_
print(f"Best parameters for final {best_model_type_name} on full data: {final_grid.best_params_}")
print(f"Best CV score for final {best_model_type_name} on full data (during tuning): {final_grid.best_score_:.3f}")

# --- Save model, scaler, and selected features list ---
print("\n--- Saving Model and Supporting Files ---")
output_dir = 'saved_models_output'
os.makedirs(output_dir, exist_ok=True)

model_filename = f"best_model_{best_model_type_name.replace(' ', '_').lower()}.joblib"
scaler_filename = f"scaler_for_{best_model_type_name.replace(' ', '_').lower()}.joblib"
features_filename = f"selected_features_{best_model_type_name.replace(' ', '_').lower()}.joblib"

model_path = os.path.join(output_dir, model_filename)
scaler_path = os.path.join(output_dir, scaler_filename)
features_path = os.path.join(output_dir, features_filename)

joblib.dump(final_best_model, model_path)
joblib.dump(final_scaler, scaler_path)
joblib.dump(selected_feature_names, features_path)

print(f"Final best model ({best_model_type_name}) saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
print(f"Selected features list saved to {features_path}")

# --- External validation ---
EXTERNAL_DATA_PATH = 'path_to_external_data.csv'
print("\n--- External Validation ---")
try:
    external_data = pd.read_csv(EXTERNAL_DATA_PATH)
except FileNotFoundError:
    print(f"Error: External data file not found at {EXTERNAL_DATA_PATH}")
    print("Skipping external validation.")
    exit()

if TARGET_COLUMN_NAME not in external_data.columns:
    print(f"Error: Target column '{TARGET_COLUMN_NAME}' not found in external data.")
    print("Skipping external validation.")
    exit()

# Load the saved model, scaler, and feature list
try:
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    loaded_selected_features = joblib.load(features_path)
except FileNotFoundError:
    print("Error: Model, scaler, or features file not found. Cannot perform external validation.")
    exit()

print(f"Model used for external validation: {best_model_type_name} (from {model_path})")

external_y_test_series = external_data[TARGET_COLUMN_NAME]

try:
    external_X_test_df = external_data[loaded_selected_features]
except KeyError as e:
    print(f"Error: One or more features used for training are missing in the external dataset: {e}")
    print("Ensure the external dataset has the following columns:", loaded_selected_features)
    exit()

external_X_test_np = external_X_test_df.values

# Scale the selected features using the LOADED scaler
external_X_test_scaled_np = loaded_scaler.transform(external_X_test_np)

# Make predictions
try:
    ext_preds_proba = loaded_model.predict_proba(external_X_test_scaled_np)[:, 1]
except Exception as e:
    print(f"Error predicting probabilities on external data: {e}")
    ext_preds_proba = np.array([0.5] * len(external_y_test_series))
    ext_auc = np.nan
else:
    try:
        ext_auc = roc_auc_score(external_y_test_series, ext_preds_proba)
    except ValueError as e:
        print(f"Could not calculate AUC for external validation: {e}")
        ext_auc = np.nan

ext_preds_binary = (ext_preds_proba > 0.5).astype(int)
cm_ext = confusion_matrix(external_y_test_series, ext_preds_binary, labels=CLASS_LABELS)
tn_ext, fp_ext, fn_ext, tp_ext = cm_ext.ravel()

ext_sensitivity = tp_ext / (tp_ext + fn_ext) if (tp_ext + fn_ext) > 0 else 0
ext_specificity = tn_ext / (tn_ext + fp_ext) if (tn_ext + fp_ext) > 0 else 0

print("\n=== External Validation Results ===")
print(f"AUC: {ext_auc:.3f}")
print(f"Sensitivity: {ext_sensitivity:.3f}")
print(f"Specificity: {ext_specificity:.3f}")
print(f"Confusion Matrix (TN, FP, FN, TP) for labels {CLASS_LABELS}: {tn_ext, fp_ext, fn_ext, tp_ext}")