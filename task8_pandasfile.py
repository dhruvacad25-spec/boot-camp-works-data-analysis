import pandas as pd


url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\n", df.head())


# Count missing values & Drop rows with missing values

print("Missing values in each column:")
print(df.isnull().sum())

df_dropped = df.dropna()
print(f"\nOriginal rows: {len(df)}")
print(f"After dropping missing rows: {len(df_dropped)}")


# Replace the missing values (with median)

df_filled = df.copy()
df_filled['total_bedrooms'] = df_filled['total_bedrooms'].fillna(df_filled['total_bedrooms'].median())
print("Missing values after replacement:")
print(df_filled.isnull().sum())


# Filter rows where median_income > 5 AND housing_median_age > 30

filtered = df_filled[(df_filled['median_income'] > 5) & (df_filled['housing_median_age'] > 30)]
print(f"Rows matching filter: {len(filtered)}")
print(filtered.head())


#  Average house value by ocean proximity

avg_by_ocean = df_filled.groupby('ocean_proximity')['median_house_value'].mean()
print(avg_by_ocean)


#  Sort dataset by median_income (Top 10 rows)

top10 = df_filled.sort_values(by='median_income', ascending=False).head(10)
print(top10[['median_income', 'median_house_value', 'ocean_proximity']])


#  Correlation between median_income and median_house_value


correlation = df_filled['median_income'].corr(df_filled['median_house_value'])
print(f"Correlation between median_income and median_house_value: {correlation:.4f}")
