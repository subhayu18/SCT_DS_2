# Task 2: Data Cleaning & EDA on Titanic Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Show first few rows
print("🔍 Sample Data:")
print(df.head())

# Basic dataset info
print("\n📄 Dataset Info:")
print(df.info())

# Check for missing values
print("\n❗ Missing Values:")
print(df.isnull().sum())

# Drop columns with too many nulls or less analytical value
df_cleaned = df.drop(['deck', 'embark_town', 'alive'], axis=1)

# Drop rows with any remaining nulls
df_cleaned.dropna(inplace=True)

# Quick stats
print("\n📊 Summary Statistics:")
print(df_cleaned.describe(include='all'))

# ----------------------------
# Visualization 1: Survival Rate by Gender
plt.figure(figsize=(6, 4))
sns.barplot(data=df_cleaned, x='sex', y='survived')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Probability')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# ----------------------------
# Visualization 2: Survival Rate by Passenger Class
plt.figure(figsize=(6, 4))
sns.barplot(data=df_cleaned, x='pclass', y='survived')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Probability')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# ----------------------------
# Visualization 3: Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_cleaned['age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# ----------------------------
# Visualization 4: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()