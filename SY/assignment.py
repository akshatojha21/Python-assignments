import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

df = pd.DataFrame({
    'Age': [24, 27, 22, 32, 29, 25, 100],
    'Salary': [50000, 60000, 52000, 58000, 62000, 55000, 600000]
})

print(df.describe())

print("Mean:", df['Age'].mean())
print("Median:", df['Age'].median())
print("Mode:", df['Age'].mode()[0])
print("Standard Deviation:", df['Age'].std())
print("Variance:", df['Age'].var())
print("Minimum:", df['Age'].min())
print("Maximum:", df['Age'].max())
print("Quantiles:", df['Age'].quantile([0.25, 0.5, 0.75]))

plt.hist(df['Age'], bins=5, edgecolor='black')
plt.title('Age Distribution')
plt.show()

sns.boxplot(y=df['Age'])
plt.title('Box Plot of Age')
plt.show()

plt.scatter(df['Age'], df['Salary'])
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

df['Age_zscore'] = zscore(df['Age'])
df['Salary_zscore'] = zscore(df['Salary'])

df_no_outliers = df[
    (df['Age_zscore'].abs() <= 3) &
    (df['Salary_zscore'].abs() <= 3)
]

print(df_no_outliers)

def cap_values(series):
    upper_limit = series.mean() + 3 * series.std()
    lower_limit = series.mean() - 3 * series.std()
    return series.clip(lower=lower_limit, upper=upper_limit)

df['Age_capped'] = cap_values(df['Age'])
df['Salary_capped'] = cap_values(df['Salary'])

print(df[['Age', 'Age_capped', 'Salary', 'Salary_capped']])
