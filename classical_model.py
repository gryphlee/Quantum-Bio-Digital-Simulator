import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model():
    # Load data
    df = pd.read_csv("data/patient_data.csv")

    # Features (X) at Target (y)
    features = ["current_glucose", "insulin_dose", "carb_intake"]
    target = "next_hour_glucose"

    X = df[features]
    y = df[target]

    # Train ang model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # I-save ang model sa isang file
    with open("patient_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved as patient_model.pkl!")

if __name__ == "__main__":
    train_model()