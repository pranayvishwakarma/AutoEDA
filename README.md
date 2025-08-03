# 📊 AutoEDA — Automated Exploratory Data Analysis Tool

AutoEDA is a full-stack web application that automates exploratory data analysis (EDA) on uploaded CSV files. It provides detailed insights, statistics, and visualizations — instantly — through an interactive web UI.

---

## 🚀 Features

- 📁 Upload any CSV file
- 📊 Automatically generates:
  - Summary statistics
  - Missing value analysis
  - Correlation matrix
  - Distribution & box plots
  - Categorical analysis
  - Outlier detection
- 🖼 Interactive charts (base64-encoded images)
- 📚 EDA history stored in MongoDB
- ⚡ Built with FastAPI + React + MongoDB

---

## 🛠 Tech Stack

| Layer       | Technology             |
|-------------|-------------------------|
| **Frontend**| React.js + Tailwind CSS |
| **Backend** | FastAPI (Python)        |
| **Database**| MongoDB (with Motor)    |
| **Charts**  | Matplotlib, Seaborn     |

---

1. **Navigate to backend**
   ```bash
   cd backend
2.pip install -r requirements.txt
3.MONGO_URL=mongodb://localhost:27017
  DB_NAME=autoeda
4.uvicorn main:app --reload

1. **Navigate to frontend**
  cd frontend
2.npm install
3.npm start

