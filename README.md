# Breast-Cancer-Detection-Using-Machine-Learning
This project aims to assist in the early detection of breast cancer using Machine Learning (ML) techniques. Early diagnosis is crucial for improving survival rates, and this project provides a simple yet effective web-based prediction app that can classify whether a breast tumor is malignant  or benign  based on input medical data.

# 💖 Breast Cancer Detection using Machine Learning  

Empowering early detection through **Machine Learning and Awareness** 🌸  

This project demonstrates how **Machine Learning** can assist in the **early diagnosis of breast cancer** by predicting whether a tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** using medical data.  

---

## 🧠 Overview  
Breast cancer is one of the most common cancers worldwide, and early detection can save lives. This project leverages the **Wisconsin Breast Cancer Diagnostic (WBCD)** dataset and builds a **Random Forest Classifier** model that classifies tumors based on several key medical features extracted from cell nuclei.

The model is deployed using **Streamlit**, providing an **interactive and visually appealing web interface** where users can enter parameters and get instant predictions.

---

## 📊 Dataset  
**Dataset Used:** [Wisconsin Breast Cancer Diagnostic Dataset (WBCD)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  

It contains 569 records with 30 numerical features describing cell characteristics such as:  
- Mean Radius  
- Mean Texture  
- Mean Perimeter  
- Mean Area  
- Mean Smoothness  
- Mean Concavity  
- Mean Symmetry  

Each record is labeled as:
- **M** → Malignant (Cancerous)  
- **B** → Benign (Non-Cancerous)

---

## ⚙️ Tech Stack  
- 🐍 **Python 3**  
- 💻 **Streamlit** — Web App Framework  
- 🤖 **Scikit-learn** — Machine Learning Library  
- 📦 **Pickle** — Model Serialization  
- 📈 **Pandas & NumPy** — Data Preprocessing  

---

## 🚀 How It Works  
1. The dataset is preprocessed and split into training and testing data.  
2. A **Random Forest Classifier** is trained and saved as `model.pkl`.  
3. The **Streamlit app** (`app.py`) loads this model.  
4. Users enter feature values (mean radius, smoothness, concavity, etc.).  
5. The app predicts whether the tumor is **Benign** or **Malignant** in real time.  

---

## 💡 Objective  
To create an easy-to-use and visually engaging app that promotes **breast cancer awareness** and showcases the potential of **Machine Learning in Healthcare**.

---

## 🖥️ Run the Project Locally  
```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-detection.git

# Navigate to the project folder
cd breast-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

Created by Rishu Raj
