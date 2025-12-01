# Multiple Disease Detection System

A Machine Learning-based web application built with Streamlit to predict the likelihood of **Diabetes**, **Heart Disease**, and **Cancer** based on user health data.

## 🚀 Features
- **Multi-Disease Prediction**: Predicts three different diseases using separate ML models.
- **User-Friendly Interface**: Simple web interface built with Streamlit.
- **Machine Learning**: Uses Logistic Regression models trained on health datasets.

## 🛠️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/rudrasharma211105/Multiple_Disease_Detection.git
   cd Multiple_Disease_Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Retrain Models**
   If you need to regenerate the model files:
   ```bash
   python diabetes.py
   python cancer.py
   python heart.py
   ```

4. **Run the Application**
   ```bash
   streamlit run frontend/app.py
   ```

## 📂 Project Structure
- `frontend/app.py`: Main Streamlit application.
- `models/`: Contains trained `.pkl` model files.
- `data/`: CSV datasets used for training.
- `*.py`: Scripts to train and save the models.
