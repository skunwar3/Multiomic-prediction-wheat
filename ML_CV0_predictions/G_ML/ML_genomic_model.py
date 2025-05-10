import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold


# Define training and evaluation for a single model
def train_and_predict(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    corr, _ = pearsonr(y_test, y_pred)

    return mape, corr


# Main pipeline for CV0 predictions with 5-fold cross-validation
def run_cv0_predictions_with_folds(X, Y, year_column, response_columns, models):
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    years = Y[year_column].unique()
    for leave_out_year in years:
        print(f"Processing year {leave_out_year}...")

        test_data = Y[Y[year_column] == leave_out_year]
        X_test = X.loc[test_data.index].values
        y_test = test_data[response_columns].values

        train_data = Y[Y[year_column] != leave_out_year]
        X_train_full = X.loc[train_data.index].values
        y_train_full = train_data[response_columns].values

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            print(f"  Fold {fold_idx + 1}/5")

            X_train = X_train_full[train_idx]
            y_train = y_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]

            for trait_idx, trait in enumerate(response_columns):
                y_train_trait = y_train[:, trait_idx]
                y_val_trait = y_val[:, trait_idx]
                y_test_trait = y_test[:, trait_idx]

                for model_name, model in models.items():
                    print(f"    Training {model_name} for {trait}...")
                    mape, corr = train_and_predict(model, X_train, y_train_trait, X_test, y_test_trait)

                    results.append({
                        'Year': leave_out_year,
                        'Trait': trait,
                        'Model': model_name,
                        'Fold': fold_idx + 1,
                        'MAPE': mape,
                        'Correlation': corr
                    })

    results_df = pd.DataFrame(results)
    results_df.to_excel('cv0_results_with_folds.xlsx', index=False)
    print("Results saved to cv0_results_with_folds.xlsx!")
    return results_df


# Example Usage
if __name__ == "__main__":
    try:
        df = pd.read_csv("G_plus_E.csv")
        X = df.iloc[:, 1:]
        Y = pd.read_csv("Y.csv")

        year_column = "Year"
        response_columns = Y.columns[2:7]

        # Define only RFR and PLSR models
        available_models = {
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'PLSRegression': PLSRegression(max_iter=200)
        }

        selected_models = ['RandomForestRegressor', 'PLSRegression']

        valid_models = {name: available_models[name] for name in selected_models if name in available_models}

        if not valid_models:
            raise ValueError("No valid models selected. Please check the model names.")

        print(f"Selected models: {', '.join(valid_models.keys())}")

        results_df = run_cv0_predictions_with_folds(X, Y, year_column, response_columns, valid_models)

        print(results_df.groupby(['Model', 'Trait']).agg({'MAPE': ['mean', 'std'], 'Correlation': ['mean', 'std']}))

    except Exception as e:
        print(f"An error occurred: {e}")
