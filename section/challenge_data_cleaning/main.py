import seaborn as sns
import pandas as pd

# 1. Load Titanic dataset
data = sns.load_dataset("titanic")

# 2. Fill missing numeric values in 'age' with the mean
data["age"].fillna(data["age"].mean(), inplace=True)

# 3. Fill missing categorical values in 'embarked' with the mode
data["embarked"].fillna(data["embarked"].mode()[0], inplace=True)

# 4. Remove duplicates
data.drop_duplicates(inplace=True)

# 5. Remove outliers in 'fare' using IQR
Q1 = data["fare"].quantile(0.25)
Q3 = data["fare"].quantile(0.75)
IQR = Q3 - Q1
cleaned_data = data[
    (data["fare"] >= Q1 - 1.5 * IQR) &
    (data["fare"] <= Q3 + 1.5 * IQR)
]

# Output cleaned dataset info
print(cleaned_data.info())