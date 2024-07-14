from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
import pandas as pd

train_df = pd.read_csv("training_data_v2_cleaned.csv")

numerical_columns = ["price", "bedroom", "bathroom", "dimensions", "built_year", "distance_to_mrt_in_m",
                     "distance_to_hawker_in_m", "distance_to_supermarket_in_m", "distance_to_sch_in_m", "distance_to_mall_in_m"]
categorical_columns = ["property_type", "furnishing",
                       "floor_level", "district_id", "tenure", "facing"]

# split data into train and test
rental_price = train_df['price']
X = train_df.drop(['price'], axis=1)

# Split the data into training and temporary sets (80% train, 20% temporary)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, rental_price, test_size=0.2, random_state=42)


class OutlierHandlerIQR(BaseEstimator, TransformerMixin):
    # Create a custom transformer class to handle outliers based on 1.5 IQR
    def fit(self, X, y):
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


# Remove outliers
outlier_handler_iqr = OutlierHandlerIQR()
X_train_new, y_train_new = outlier_handler_iqr.fit_transform(X_train, y_train)

# Scale numerical columns and encode categorical columns
column_transformer = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), [
         col for col in numerical_columns if col != "price"]),
        ("encoder", OneHotEncoder(drop=None, sparse_output=False), categorical_columns)
    ],
    remainder="passthrough"  # Include the boolean columns without transformation
)

X_train_new = column_transformer.fit_transform(X_train_new)

catboost = CatBoostRegressor()

# Define hyperparameter grid
cb_param_grid = {
    "iterations": [1000],
    "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1],
    "depth": [6, 7, 8, 9, 10],
    "subsample": [0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
    "colsample_bylevel": [0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
    "min_data_in_leaf": [20, 40, 60, 80, 100],
}

# Create a grid search object
cb_grid_search = GridSearchCV(estimator=catboost,
                              param_grid=cb_param_grid,
                              cv=5,
                              n_jobs=-1,
                              scoring="neg_root_mean_squared_error",
                              verbose=4)

# Fit the grid search to the training data
cb_grid_search.fit(X_train_new, y_train_new)

print(cb_grid_search.best_params_)
print(cb_grid_search.best_score_)
