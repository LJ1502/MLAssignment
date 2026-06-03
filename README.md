# 🧞‍♂️ The Dating Profile Akinator

This project is a Machine Learning web application built with Python and Streamlit. It uses a trained Decision Tree model to predict the "popularity" of a dating profile based on physical demographics, app usage habits, and profile content. 

The application features a **Fault-Tolerant AI Architecture**. It can use the Google Gemini API to dynamically extract data from a user's bio, but gracefully falls back to mathematical averages (Offline Mode) if an API key is not provided or token limits are reached.

## 📁 Project Structure
Ensure you have the following files in the same folder before running the app:
* `app.py` - The main Streamlit web application.
* `dating_model.pkl` - The pre-trained Scikit-Learn Decision Tree model.
* `MachineLearningModel.ipynb` - The Jupyter Notebook used for data cleaning, feature extraction, and model training.

---

## ⚙️ Installation & Setup

**1. Prerequisites**
You must have Python installed on your computer (Python 3.8 or higher is recommended). 

**2. Install Required Modules**
Open your computer's Terminal or Command Prompt, navigate to this project folder, and run the following command to install all required dependencies:

```bash
pip install streamlit pandas joblib google-generativeai
