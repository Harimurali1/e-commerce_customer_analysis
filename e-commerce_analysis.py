import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv("customers.csv")

# Exploratory Data Analysis (EDA)
print("First five rows of the data:")
print(customers.head())

print("\nData Summary:")
print(customers.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(customers.isnull().sum())

# Visualization 1: Distribution of Annual Income
plt.figure(figsize=(8, 5))
plt.hist(customers['Annual_Income'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Visualization 2: Spending Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(customers['Spending_Score'], bins=20, kde=True, color='green')
plt.title('Distribution of Spending Score')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Visualization 3: Relationship between Age and Annual Income
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual_Income', y='Age', data=customers, color='purple')
plt.title('Annual Income vs Age')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Age')
plt.grid(True)
plt.show()

# Visualization 4: Boxplot of Annual Income by Gender
plt.figure(figsize=(8, 5))
sns.boxplot(x='Gender', y='Annual_Income', data=customers, palette='Set2')
plt.title('Annual Income Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Annual Income (k$)')
plt.grid(True)
plt.show()

# Visualization 5: Pairplot to explore correlations
sns.pairplot(customers[['Annual_Income', 'Spending_Score', 'Age']])
plt.suptitle('Pairplot of Income, Spending Score, and Age', y=1.02)
plt.show()

# Visualization 6: Heatmap of Correlations
plt.figure(figsize=(8, 6))
sns.heatmap(customers.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualization 7: Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(customers['Age'], bins=20, color='orange', kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Visualization 8: Spending Score vs Annual Income
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Gender', data=customers, palette='coolwarm')
plt.title('Spending Score vs Annual Income by Gender')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show()
