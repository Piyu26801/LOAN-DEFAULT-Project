import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading data...")
try:
    loanData = pd.read_csv('cleanData.csv')
    X = loanData.drop('Default', axis=1)
    
    # We only need to scale numerical columns.
    # But since the previous pipeline might have scaled all columns including one-hot encoded, 
    # we just fit on X exactly as it was.
    print("Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    print("Saving scaler...")
    joblib.dump(scaler, 'scaler.pkl')
    print("Done! scaler.pkl created successfully.")
except Exception as e:
    print(f"Error: {e}")
