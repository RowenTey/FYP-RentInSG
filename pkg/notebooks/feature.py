import pandas as pd
import matplotlib.pyplot as plt
import pickle

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

with open("../streamlit/static/column_transformer.pkl", "rb") as file:
    column_transformer = pickle.load(file)

# Load catboost model from pickle file
with open("../streamlit/static/catboost.pkl", "rb") as file:
    model = pickle.load(file)

feature_importances = model.feature_importances_

# Get feature names after transformation
num_features = column_transformer.named_transformers_[
    'scaler'].get_feature_names_out([col for col in numerical_columns if col != "price"])
cat_features = column_transformer.named_transformers_[
    'encoder'].get_feature_names_out(categorical_columns)
feature_names = list(num_features) + list(cat_features) + \
    ["is_whole_unit", "has_pool", "has_gym"]

# Create a DataFrame with feature names and importances
feature_importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
})

# Programmatically create the mapping for one-hot encoded features
feature_mappings = {}

for feature in categorical_columns:
    encoded_features = [col for col in cat_features if col.startswith(feature)]
    feature_mappings[feature] = encoded_features

# Initialize a dictionary to store combined importances
combined_importances = {}

# Loop through the mappings and sum importances
for original_feature, encoded_features in feature_mappings.items():
    combined_importances[original_feature] = feature_importances_df[feature_importances_df['feature'].isin(
        encoded_features)]['importance'].sum()

# For numerical features, directly add their importances
for feature in list(num_features) + ['is_whole_unit', 'has_pool', 'has_gym']:
    combined_importances[feature] = feature_importances_df[feature_importances_df['feature']
                                                           == feature]['importance'].values[0]

# Convert the combined importances to a DataFrame for easy visualization
combined_importances_df = pd.DataFrame(
    list(combined_importances.items()), columns=['feature', 'importance'])

print(combined_importances_df)

combined_importances_df.to_csv('static/feature_importance.csv', index=False)

# Visualize the combined feature importances
plt.figure(figsize=(10, 6))
plt.barh(width='importance', y='feature', data=combined_importances_df.sort_values(
    by='importance', ascending=False))
plt.title('Rental Price Prediction Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
