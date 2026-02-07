import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
data = sns.load_dataset("titanic")

# 2. Fill missing values
data["age"].fillna(___, inplace=True)
data["embarked"].fillna(___, inplace=True)

# 3. Encode 'sex' and 'embarked' using pd.get_dummies()
data = pd.get_dummies(data, columns=["___", "___"], drop_first=True)

# 4. Scale 'age' and 'fare' using StandardScaler
scaler = StandardScaler()
data[["age_scaled", "fare_scaled"]] = scaler.fit_transform(data[["___", "___"]])

# 5. Create new column 'family_size'
data["family_size"] = ___

# Save transformed dataset
transformed_data = data.copy()

print(transformed_data.head())