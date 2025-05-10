import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr

# Random Forest Regressor
def train_random_forest(X, y, k_fold, n_estimators):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    mape_list = []
    corr_list = []
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mape_list.append(mean_absolute_percentage_error(y_test, y_pred))
        corr_list.append(pearsonr(y_test, y_pred)[0])

    return model, mape_list, corr_list

# Partial Least Squares Regression
def train_pls_regression(X, y, k_fold, max_iter):
    model = PLSRegression(max_iter=max_iter)
    mape_list = []
    corr_list = []
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).flatten()

        mape_list.append(mean_absolute_percentage_error(y_test, y_pred))
        corr_list.append(pearsonr(y_test, y_pred)[0])

    return model, mape_list, corr_list

# Main Pipeline
def run_pipeline(X, Y, selected_models, response_columns, k_fold=5, n_estimators=100, max_iter=200):
    mape_results = []
    corr_results = []

    # Reset index to ensure consistency
    X = pd.DataFrame(X).reset_index(drop=True)
    Y = pd.DataFrame(Y).reset_index(drop=True)

    for trait in response_columns:
        y = Y[trait].values
        print(f"\n\nStarting optimization for trait: {trait}")

        for model_name in selected_models:
            if model_name == 'RandomForestRegressor':
                _, mape, corr = train_random_forest(X.values, y, k_fold, n_estimators)
            elif model_name == 'PLSRegression':
                _, mape, corr = train_pls_regression(X.values, y, k_fold, max_iter)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            mape_results.append(pd.DataFrame({'Model': [model_name] * len(mape),
                                              'Trait': [trait] * len(mape),
                                              'Iteration': range(1, len(mape) + 1),
                                              'MAPE': mape}))
            corr_results.append(pd.DataFrame({'Model': [model_name] * len(corr),
                                              'Trait': [trait] * len(corr),
                                              'Iteration': range(1, len(corr) + 1),
                                              'Correlation': corr}))

    mape_df = pd.concat(mape_results, ignore_index=True)
    corr_df = pd.concat(corr_results, ignore_index=True)

    mape_summary = mape_df.groupby(['Model', 'Trait']).agg({'MAPE': ['mean', 'std']}).reset_index()
    corr_summary = corr_df.groupby(['Model', 'Trait']).agg({'Correlation': ['mean', 'std']}).reset_index()

    mape_df.to_excel('mape_results.xlsx', index=False)
    corr_df.to_excel('correlation_results.xlsx', index=False)

    print("\n\n################# MAPE Summary #################\n")
    print(mape_summary)
    print("\n\n################# Correlation Summary #################\n")
    print(corr_summary)

    return mape_summary, corr_summary

# Example Usage
if __name__ == "__main__":
    try:
        df = pd.read_csv("G_H_W.csv")
        X = df.iloc[:, 1:]
        Y = pd.read_csv("Y.csv")

        selected_models = ['RandomForestRegressor', 'PLSRegression']
        response_columns = Y.columns[2:7]

        run_pipeline(X, Y, selected_models, response_columns)
    except Exception as e:
        print(f"An error occurred: {e}")
