# Titanic Survival Prediction — AI Assignment 2
## Project Overview
- This project analyzes the Titanic dataset to identify the key factors that influenced passenger survival. - By performing rigorous Data Cleaning, Feature Engineering, and Feature Selection, we transformed raw historical data into a structured, model-ready format for machine learning.
## Project Structure
The project is organized into a modular structure to separate data, exploration, and reusable logic:
```text
titanic_assignment/
├── data/
│   ├── train.csv               # Raw Training Data
│   ├── test.csv                # Raw Test Data
│   └── train_cleaned.csv       # Final Processed Data
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb  # Primary Analysis
├── scripts/
│   ├── data_cleaning.py        # Cleaning functions
│   ├── feature_engineering.py   # Transformation logic
│   └── feature_selection.py    # Importance & Correlation logic
├── README.md                   # Project Documentation
└── requirements.txt            # Python Dependencies
```
### Data Cleaning & Decisions
The following steps were taken to handle missing data and ensure consistency:
- Age: Missing values were imputed using the Median ($28.0$) to mitigate the impact of outliers.
- Embarked: Missing values were filled with the Mode ('S'), the most frequent port.
- Cabin: Dropped the column due to a high missing value rate ($>70\%$), which prevents reliable inference.
- Outliers: Applied a Log Transformation to Fare to handle extreme values and right-skewness.
- Consistency: Verified and standardized formatting for Sex and Pclass.
### Feature Engineering
To improve predictive power, several derived features were created:
- FeatureDescriptionFamilySizeCombined SibSp + Parch + 1 (total family members).IsAloneBinary flag (1 if traveling alone, 0 otherwise).
- TitleExtracted from Name and grouped (Mr, Mrs, Miss, Master, Rare).
- AgeGroupCategorized into life stages (Child, Teen, Adult, Senior).
- Fare_LogLog-transformed fare to normalize distribution.
- EncodingApplied One-Hot Encoding to nominal variables (Sex, Embarked, Title, AgeGroup).
### Feature Selection
We refined the feature set using two primary methods:
- Correlation Analysis: Generated a heatmap to identify and remove redundant features (e.g., keeping Fare_Log while dropping the raw Fare).
- Random Forest Importance: Used a tree-based model to rank features.
- Final Selection: Sex_male, Title_Mr, Age, and Pclass were identified as the top predictive variables.
### Key Findings
- Gender & Title: 'Sex' and 'Title' (specifically 'Mr') were the most significant predictors of survival.
- Social Class: Pclass (1st Class) showed a strong positive correlation with survival rates.
- Signal Strength: Engineered features like FamilySize and Fare_Log provided a cleaner signal than the original raw inputs.

## How to Run the Project
Clone the repository:
```
git clone <your-repo-link>
```
cd titanic_assignment
### Setup (Windows + VS Code)

1. Open the project folder in VS Code.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not available:
   ```powershell
   pip install pandas matplotlib seaborn scikit-learn
   ```
### Technologies Used
- Python 3.10+ recommended
- Pandas & NumPy: Data manipulation and numerical operations.
- Matplotlib & Seaborn: Statistical data visualization.
-Scikit-learn: Feature ranking and data preprocessing.









