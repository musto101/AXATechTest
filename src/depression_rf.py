# Re-import necessary libraries and re-establish context after system interruption
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

# Reload the data
file_path = 'data/depression/depression_data.csv'
data = pd.read_csv(file_path)

# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    if column != 'History of Mental Illness':  # Target variable handled separately
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Encode the target variable (History of Mental Illness)
target_encoder = LabelEncoder()
data['History of Mental Illness'] = target_encoder.fit_transform(data['History of Mental Illness'])

# Separate features (X) and target variable (y)
X = data.drop(['History of Mental Illness', 'Name'], axis=1)  # Drop Name as it's not a feature
y = data['History of Mental Illness']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardise the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict_proba(X_test)

# Evaluate the model using ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred[:, 1])

# Display the classification report and confusion matrix
print(f"ROC AUC Score: {roc_auc}")

