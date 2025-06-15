import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# FEATURE ENGINEERING
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 1
df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

# Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr','Major',
                                   'Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Drop unneeded columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Convert to numeric
df['Age'] = pd.to_numeric(df['Age'])
df['Fare'] = pd.to_numeric(df['Fare'])

# Create AgeGroup and FareBin
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 60, 100], labels=['Child', 'Adult', 'Senior'])
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[1, 2, 3, 4])

# Encode categorical features
le = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title', 'AgeGroup']:
    df[col] = le.fit_transform(df[col])

# Drop original Age and Fare
df.drop(['Age', 'Fare'], axis=1, inplace=True)

# Define features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, rf_preds))
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, rf_preds))

# PLOT: Survival Rate by Family Size
plt.figure(figsize=(8, 6))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("family_size_survival_plot.png")
plt.show()

