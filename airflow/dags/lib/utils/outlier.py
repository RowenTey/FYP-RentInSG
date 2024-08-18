from sklearn.base import BaseEstimator, TransformerMixin

class OutlierHandler2Point5SD(BaseEstimator, TransformerMixin):
    def fit(self, _, y):
        # Calculate mean, standard deviation, and cutoff values of target label (y)
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
        print(f"Rental price outliers based on 3 SD: {y.shape[0] - y[mask].shape[0]}")
        # Return data with outliers removed 
        return X[mask], y[mask]

    def fit_transform(self, X, y):
        # Perform both fit and transform 
        return self.fit(X, y).transform(X, y)
    
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
        print(f"Rental price outliers based on 1.5 IQR: {y.shape[0] - y[mask].shape[0]}")
        # Return data with outliers removed 
        return X[mask], y[mask]

    def fit_transform(self, X, y):
        # Perform both fit and transform
        return self.fit(X, y).transform(X, y)

