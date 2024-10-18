def remove_outliers(df, key_col):
    # Remove outliers using the IQR method based on key_col
    Q1 = df[key_col].quantile(0.25)
    Q3 = df[key_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Keep only data points within the lower and upper bounds
    return df[(df[key_col] >= lower_bound) & (df[key_col] <= upper_bound)]