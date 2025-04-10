# Admission_Predictor_application
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://uclaneuralnetworkapplutionwithapp-main-9evbc85xy45t9gmmi7w9di.streamlit.app/)

password - streamlit

This application predicts your chances of being admitted to a university program based on academic performance and other key indicators. The model is trained on a dataset of past admission records and helps visualize your admission probability.

## Features
- Clean and interactive Streamlit interface.
- Form-based input for entering:
   - GRE and TOEFL scores
   - CGPA
   - SOP & LOR strength
   - University rating
   - Research experience

- Instant prediction of admission chances (expressed as a percentage).
- Visualizations to understand feature importance (e.g., CGPA vs. admission rate, training loss curve).
- Fully deployed on the Streamlit Community Cloud.

## Dataset
The model is trained on the Processed Admission Dataset, which includes the following features:
- GRE Score (out of 340)
- TOEFL Score (out of 120)
- Statement of Purpose (SOP) and Letter of Recommendation (LOR) strength (1–5 scale)
- CGPA (on a 10-point scale)
- University Rating (1 to 5)
- Research Experience (binary: Yes/No)

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model
We use a Multilayer Perceptron (MLP) Neural Network trained on admission data. The model incorporates:
- One-hot encoding of categorical features (e.g., University Rating and Research).
- Normalized numerical inputs (e.g., CGPA, GRE, TOEFL).
- Output: Probability of admission (between 0 and 1).

## Future Enhancements
* Include more features (like university name or major).
* Add model explainability with SHAP or LIME.
* Allow batch prediction for multiple applicants.
* Enable saving of prediction history for user comparison.edictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/mariemeba123/UCLA_Neural_Networks_Solution_with_Streamlit-main.git
   cd UCLA_Neural_Networks_Solution_with_Streamlit-main

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the Admission Predictor Application! Feel free to share your feedback.
