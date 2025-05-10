#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:19:49 2025

@author: sudipkunwar
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import pearsonr

# Define your training functions

def train_random_forest(X, y, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

def train_pls_regression(X, y, max_iter=200):
    model = PLSRegression(max_iter=max_iter)
    model.fit(X, y)
    return model

# Main pipeline for forward predictions using new CV column (independent dataset)
def run_predictions_with_folds(X, Y, scheme_column, response_columns, models, k_fold=5):
    results = []

    # Define training and testing datasets based on the CV scheme
    train_data = Y[Y[scheme_column] == 0]
    test_data = Y[Y[scheme_column] == 1]

    X_train_full = X.loc[train_data.index].values
    y_train_full = train_data[response_columns].values
    X_test = X.loc[test_data.index].values
    y_test = test_data[response_columns].values

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    for trait_idx, trait in enumerate(response_columns):
        y_train_trait = y_train_full[:, trait_idx]
        y_test_trait = y_test[:, trait_idx]

        for model_name, model_train_func in models.items():
            print(f"  Training {model_name} for {trait}...")

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
                X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
                y_train_fold = y_train_trait[train_idx]

                if model_name == 'RandomForest':
                    model = train_random_forest(X_train, y_train_fold)
                elif model_name == 'PLSRegression':
                    model = train_pls_regression(X_train, y_train_fold)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                y_pred_test = model.predict(X_test)
                mape_test = mean_absolute_percentage_error(y_test_trait, y_pred_test)
                corr_test = pearsonr(y_test_trait, y_pred_test)[0]

                results.append({
                    'Trait': trait,
                    'Model': model_name,
                    'Fold': fold_idx + 1,
                    'MAPE': mape_test,
                    'Pearson Correlation': corr_test
                })

    results_df = pd.DataFrame(results)
    results_df.to_excel('Independent_prediction_results_with_folds.xlsx', index=False)
    print("Results saved to Independent_prediction_results_with_folds.xlsx!")
    return results_df

# Example Usage
if __name__ == "__main__":
    try:
        df = pd.read_csv("H_plus_E.csv")
        X = df.iloc[:, 1:]
        Y = pd.read_csv("Y.csv")

        scheme_column = "CV"
        response_columns = Y.columns[2:7]

        available_models = {
            'RandomForest': train_random_forest,
            'PLSRegression': train_pls_regression
        }

        selected_models = ['RandomForest', 'PLSRegression']  # Choose the models you want to use
        valid_models = {name: available_models[name] for name in selected_models if name in available_models}

        if not valid_models:
            raise ValueError("No valid models selected. Please check the model names.")

        print(f"Selected models: {', '.join(valid_models.keys())}")

        results_df = run_predictions_with_folds(X, Y, scheme_column, response_columns, valid_models)

        # Print summary
        print(results_df.groupby(['Model', 'Trait']).agg({'MAPE': ['mean', 'std'], 'Pearson Correlation': ['mean', 'std']}))

    except Exception as e:
        print(f"An error occurred: {e}")
