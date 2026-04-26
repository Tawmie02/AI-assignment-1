import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def perform_feature_selection(df, target_col='Survived'):
    """
    Performs Correlation Analysis and Random Forest Feature Importance.
    """
    # 1. Prepare data (Drop non-predictive and redundant columns)
    # We drop 'Fare' because we use 'Fare_Log'
    # We drop 'FamilySize' if we are using 'SibSp' and 'Parch' to avoid multi-collinearity
    drop_list = [target_col, 'PassengerId', 'Name', 'Ticket', 'Fare', 'FamilySize']
    X = df.drop(columns=[c for c in drop_list if c in df.columns])
    y = df[target_col]

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    correlation = X.corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # 3. Feature Importance using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Create Importance Table
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # 5. Plot Top Features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title("Top 10 Selected Features")
    plt.show()

    return importance_df

if __name__ == "__main__":
    # Load the cleaned data generated from your notebook
    try:
        cleaned_data = pd.read_csv('../data/train_cleaned.csv')
        rankings = perform_feature_selection(cleaned_data)
        
        print("\n--- Final Selected Features ---")
        print(rankings)
        
    except FileNotFoundError:
        print("Error: train_cleaned.csv not found. Run the notebook or cleaning script first.")