import optuna
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

train_df = pd.read_csv("training_data_v2_cleaned.csv")

rental_price = train_df['price']
X = train_df.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, rental_price, test_size=0.2)

numerical_columns = [
    "price",
    "bedroom",
    "bathroom",
    "dimensions",
    "built_year",
    "distance_to_mrt_in_m",
    "distance_to_hawker_in_m",
    "distance_to_supermarket_in_m",
    "distance_to_sch_in_m",
    "distance_to_mall_in_m"]
categorical_columns = ["property_type", "furnishing",
                       "floor_level", "district_id", "tenure", "facing"]

column_transformer = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), [
         col for col in numerical_columns if col != "price"]),
        ("encoder", OneHotEncoder(drop=None, sparse_output=False), categorical_columns)
    ],
    remainder="passthrough"  # Include the boolean columns without transformation
)


class OutlierHandlerIQR(BaseEstimator, TransformerMixin):
    def fit(self, _, y):
        # Calculate quartiles, IQR and cutoff values of target label (y)
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_cutoff = Q1 - 1.5 * IQR
        self.upper_cutoff = Q3 + 1.5 * IQR
        print(f"Lower cutoff: {round(self.lower_cutoff)} S$/month")
        print(f"Upper cutoff: {round(self.upper_cutoff)} S$/month")
        return self

    def transform(self, X, y):
        # Apply cutoff values
        mask = (y >= self.lower_cutoff) & (y <= self.upper_cutoff)
        # Print number of outliers
        print(
            f"Rental price outliers based on 1.5 IQR: {y.shape[0] - y[mask].shape[0]}")
        # Return data with outliers removed
        return X[mask], y[mask]

    def fit_transform(self, X, y):
        # Perform both fit and transform
        return self.fit(X, y).transform(X, y)


class OutlierHandlerSD(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        # Calculate mean, standard deviation, and cutoff values of target label
        # (y)
        self.mean = y.mean()
        self.sd = y.std()
        self.lower_cutoff = self.mean - 2.5 * self.sd
        self.upper_cutoff = self.mean + 2.5 * self.sd
        print(f"Lower cutoff: {round(self.lower_cutoff)} S$/month")
        print(f"Upper cutoff: {round(self.upper_cutoff)} S$/month")
        return self

    def transform(self, X, y):
        # Apply cutoff values
        mask = (y >= self.lower_cutoff) & (y <= self.upper_cutoff)
        # Print number of outliers
        print(
            f"Rental price outliers based on 2.5 SD: {y.shape[0] - y[mask].shape[0]}")
        # Return data with outliers removed
        return X[mask], y[mask]

    def fit_transform(self, X, y):
        # Perform both fit and transform
        return self.fit(X, y).transform(X, y)


# the use of fit is to find the mean and variance
X_train_1 = column_transformer.fit_transform(X_train)

# For the test dataset, you do not need to use fit again, as we are using
# the mean and variance from the train dataset
X_test_1 = column_transformer.transform(X_test)

handler = OutlierHandlerIQR()
# handler = OutlierHandlerSD()
X_train_1, y_train = handler.fit_transform(X_train_1, y_train)
X_test_1, y_test = handler.transform(X_test_1, y_test)


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 3000),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100), }

    model = cb.CatBoostRegressor(**params, silent=True)
    model.fit(X_train_1, y_train)
    predictions = model.predict(X_test_1)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)


# def objective2(trial):
#     print("Optimizing hyperparameters: ", trial)
#     params = {
#         "iterations": 1000,
#         "learning_rate": trial.suggest_float("learning_rate", 0.08, 0.1, log=True),
#         "depth": trial.suggest_int("depth", 5, 15),
#         "subsample": trial.suggest_float("subsample", 0.4, 1.0),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.8, 1.0),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
#     }

#     model = cb.CatBoostRegressor(**params, silent=True)
#     model.fit(X_train_1, y_train)
#     predictions = model.predict(X_test_1)
#     rmse = mean_squared_error(y_test, predictions, squared=False)
#     return rmse


# study2 = optuna.create_study(direction='minimize')
# study2.optimize(objective2, n_trials=30)

# print('Best hyperparameters:', study2.best_params)
# print('Best RMSE:', study2.best_value)


# def catboost(X_train, y_train, X_test, y_test, params):
#     model = cb.CatBoostRegressor(**params, silent=True)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     metrics = {}
#     metrics["rmse"] = mean_squared_error(y_test, predictions, squared=False)
#     metrics["mae"] = mean_absolute_error(y_test, predictions)
#     metrics["accuracy"] = explained_variance_score(y_test, predictions)
#     return metrics


# # print(catboost(X_train_1, y_train, X_test_1, y_test, {'iterations': 1000, 'learning_rate': 0.08981971475423699,
# #       'depth': 9, 'subsample': 0.429612270456331, 'colsample_bylevel': 0.9962717560011394, 'min_data_in_leaf': 14}))
# print(catboost(X_train_1, y_train, X_test_1, y_test, {'iterations': 1000, 'learning_rate': 0.09625671943145078, 'depth': 10, # noqa: E501
#       'subsample': 0.9032672919879902, 'colsample_bylevel': 0.8031720441028976, 'min_data_in_leaf': 92}))
