# cardiovascular-disease-prediction-using-machine-learning
Here's a README description for your GitHub repository:  

---

# Cardiovascular Disease Prediction  

## Overview  
This project is a **machine learning-based web application** that predicts the likelihood and severity of cardiovascular disease based on user-inputted health parameters. The model assists doctors and individuals in assessing heart disease risk, particularly in **rural and remote areas** where access to advanced diagnostic tools is limited.  

The application is built using **Streamlit** for the frontend and **Scikit-Learn** for the backend machine learning models.  

## Features  
- Accepts key health parameters such as age, blood pressure, cholesterol levels, glucose levels, and lifestyle habits.  
- Utilizes **three machine learning models**:  
  - **Logistic Regression (SGD Classifier)**  
  - **Random Forest Classifier**  
  - **Gradient Boosting Classifier**  
- Provides **risk probability (%)** and classifies the user's cardiovascular health into different **stages** (Healthy, Stage 1, Stage 2, Stage 3).  
- Offers a **visual comparison** of model predictions using bar charts.  

## Technologies Used  
- **Python** (NumPy, Pandas, Scikit-Learn)  
- **Streamlit** (for building the interactive web app)  
- **Matplotlib** (for visualization)  

## How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/cardiovascular-disease-prediction.git
   cd cardiovascular-disease-prediction
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

## Dataset  
The model is trained on a **preprocessed cardiovascular disease dataset** (`processed_data.csv`). The dataset includes patient information such as **age, height, weight, blood pressure, cholesterol, glucose levels, smoking habits, alcohol consumption, and physical activity**.  

## Future Improvements  
- Integration of **deep learning models** for improved accuracy.  
- Addition of **real-time patient monitoring** features.  
- Deployment on **cloud platforms** for broader accessibility.  

---

Let me know if you want any modifications! 🚀
