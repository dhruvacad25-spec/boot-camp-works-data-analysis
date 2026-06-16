import pandas as pd


#  Employee DataFrame

# Create the DataFrame with 5+ rows of employee data
data = {
    'Name':       ['Alice',   'Bob',     'Charlie', 'Diana',   'Eve',     'Frank'],
    'Age':        [28,         35,        42,        31,        26,        38],
    'Salary':     [55000,      72000,     85000,     60000,     48000,     91000],
    'Department': ['HR',       'Finance', 'IT',      'HR',      'Finance', 'IT'],
    'Experience': [3,          8,         15,        5,         2,         12]
}

df = pd.DataFrame(data)


print(df)


#  Already done above (5+ rows added)



#  Sort by Department (ascending) then by Salary (descending)

df_sorted = df.sort_values(
    by=['Department', 'Salary'],
    ascending=[True, False]
)

print("\n--- Sorted DataFrame (Department ↑, Salary ↓) ---")
print(df_sorted)


#  Display top 5 rows of sorted DataFrame

print("\n--- Top 5 Rows of Sorted DataFrame ---")
print(df_sorted.head(5))
