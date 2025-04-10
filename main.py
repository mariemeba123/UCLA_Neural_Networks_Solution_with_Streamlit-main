# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Swapnil Kangralkar',
#     license='',
# )

from src.data.make_dataset import load_data
from src.visualization.visualize import plot_Loss_Curve, plot_gpA
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_MLPmodel
from src.models.predict_model import evaluate_model
import pandas as pd
if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/Admission(in).csv"
    data = load_data(data_path)

    # Create dummy variables and separate features and target
    x, y = create_dummy_vars(data)

    # Train the logistic regression model
    MLP, xtest_scaled, ytest = train_MLPmodel(x, y)

    # Evaluate the model
    plot_Loss_Curve(MLP)
    # Load the dataset
    csv_path = "data/processed/Processed_Admission_Dataset.csv"
    df = pd.read_csv(csv_path)
    plot_gpA(df)
    accuracy, confusion_mat = evaluate_model(MLP, xtest_scaled, ytest)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")
