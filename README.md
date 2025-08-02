# Glucose Level Prediction using the Framingham Dataset

This project applies machine learning techniques to predict blood glucose levels based on health indicators from the Framingham Heart Study dataset. It explores data preprocessing, feature engineering, and regression modeling.

---

## 📌 Project Objectives

- Import and explore the dataset
- Clean and preprocess missing or inconsistent data
- Visualize key variables related to glucose
- Select relevant features for prediction
- Train and evaluate multiple regression models
- Identify the most influential factors affecting glucose levels
- Generate insights for public health relevance

---

## 📂 Dataset

- **Name**: `framingham.csv`
- **Source**: Public Framingham Heart Study (mock version used here)
- **Target Variable**: `glucose`

---

## 🔍 Models Used

| Model               | Mean Squared Error (MSE) | R² Score |
|--------------------|--------------------------|----------|
| Linear Regression  | 0.748                    | 0.486    |
| Random Forest      | 0.723 ✅                 | 0.504 ✅ |

- Random Forest performed better in both metrics.

---

## 📈 Key Insights

- **Top influencing features**: BMI, total cholesterol, heart rate, age
- Random Forest model explains ~50% of glucose variation — useful for early risk detection
- Models suggest a moderate but reliable association between lifestyle/health metrics and glucose levels

---

## 🛠 Technologies Used

- Google Colab (Python)
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Machine Learning)

---

## 📌 How to Run

1. Clone the repository or download the files
2. Open the `Glucose_Prediction_Summary.ipynb` notebook
3. Run the cells in order (or use Google Colab)
4. Ensure `framingham.csv` is in the same directory

---

## 📚 Future Improvements

- Add classification models (e.g., diabetes risk classification)
- Hyperparameter tuning for better performance
- Integrate external health or lifestyle datasets

---

## 👩‍💻 Author

**Anya Kumar**  
B.Tech in CSE (AI/ML Specialization)  
*Project for academic learning and skill development*

---

