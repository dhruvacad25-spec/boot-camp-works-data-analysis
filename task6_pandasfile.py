import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, 40, 45],
    'Salary':[50000,60000,70000,80000,90000],
    'Department':['HR','Finance','Tech','Tech','HR'],
    'Experience':[2,5,8,10,12]
}
df=pd.DataFrame(data)
filtered_df=df[(df['Salary']>60000) & (df['Experience']>5) & (df['Department']=='Tech')]
print(filtered_df[['Name','Salary','Experience']])
