import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess data
df = pd.read_csv('fake_accounts.csv')
X = df.drop(['Username', 'Label'], axis=1)
X['Profile Picture Present (Yes/No)'] = X['Profile Picture Present (Yes/No)'].map({'Yes': 1, 'No': 0})
y = df['Label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'fake_account_detector.joblib')