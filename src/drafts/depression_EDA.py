import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data/depression/depression_data.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Summary statistics for numerical columns
summary_stats = data.describe()
print("Summary Statistics for Numerical Data:")
print(summary_stats)

# Categorical data overview
categorical_overview = {col: data[col].unique() for col in data.select_dtypes(include='object').columns}
print("\nUnique Values in Categorical Columns:")
for col, values in categorical_overview.items():
    print(f"{col}: {values}")

# Function to create a bar chart for categorical data
def plot_categorical_distribution(column, title):
    data[column].value_counts().plot(kind='bar', figsize=(8, 6))
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    # save the plot
    plt.savefig('data/figures/' + column + '.png')

# Function to plot numerical data distributions
def plot_numerical_distribution(column, title):
    data[column].plot(kind='hist', bins=20, figsize=(8, 6), alpha=0.7)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    # save the plot
    plt.savefig('data/figures/' + column + '.png')

# Plot distributions of some key categorical columns
plot_categorical_distribution('Marital Status', 'Distribution of Marital Status')
plot_categorical_distribution('Smoking Status', 'Distribution of Smoking Status')
plot_categorical_distribution('Alcohol Consumption', 'Distribution of Alcohol Consumption')

# Plot numerical data distributions
plot_numerical_distribution('Age', 'Distribution of Age')
plot_numerical_distribution('Income', 'Distribution of Income')

# Analyzing relationships: Family history of depression and history of mental illness
family_history_vs_mental_illness = data.groupby('Family History of Depression')['History of Mental Illness'].value_counts(normalize=True).unstack()
family_history_vs_mental_illness.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Mental Illness Based on Family History of Depression')
plt.ylabel('Proportion')
plt.savefig('data/figures/family_history_vs_mental_illness.png')

# Analyzing the relationship between alcohol consumption and mental illness
alcohol_vs_mental_illness = data.groupby('Alcohol Consumption')['History of Mental Illness'].value_counts(normalize=True).unstack()
alcohol_vs_mental_illness.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Mental Illness Based on Alcohol Consumption')
plt.ylabel('Proportion')
plt.savefig('data/figures/alcohol_vs_mental_illness.png')
