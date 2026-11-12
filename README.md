# 🌱 AgriSense AI – Hybrid Crop Advisor

AgriSense AI is a machine learning-based crop recommendation system that helps users identify suitable crops based on soil nutrients and weather conditions.

The system uses agricultural parameters to generate crop recommendations through a trained machine learning model.

## ✨ Features

- Crop recommendations based on soil and weather parameters
- Soil nutrient inputs: Nitrogen (N), Phosphorus (P), and Potassium (K)
- Temperature, humidity, pH and rainfall inputs
- Machine learning-based prediction
- Backend API for processing prediction requests
- Deployed backend API

## 🛠️ Tech Stack

- **Frontend:** React
- **Backend:** Python, FastAPI
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Storage:** Joblib
- **Deployment:** Vercel

## 📁 Project Structure

```text
agrisense-ai/
├── api/
│   └── index.py
├── data/
│   └── Crop_recommendation.csv
├── models/
│   ├── crop_model.joblib
│   └── metrics.json
├── main.py
├── train.py
├── requirements.txt
└── vercel.json
```

## ⚙️ Input Parameters

The model uses seven input parameters:

| Parameter | Description |
|---|---|
| Nitrogen (N) | Nitrogen content in soil |
| Phosphorus (P) | Phosphorus content in soil |
| Potassium (K) | Potassium content in soil |
| Temperature | Environmental temperature |
| Humidity | Relative humidity |
| pH | Soil pH level |
| Rainfall | Rainfall measurement |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/charitha0310/agrisense-ai.git
cd agrisense-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python main.py
```

Check the actual application entry point and run instructions before relying on this command.

## 📊 Model Evaluation

Model evaluation results are stored in `models/metrics.json`.

## 🌐 Deployment

Backend API: https://agrisense-backend-three.vercel.app

## 🔮 Future Improvements

- Improve prediction performance through model experimentation
- Add more regional agricultural data
- Enhance the user interface and user experience
- Introduce additional crop insights

## 👩‍💻 Author

**Rasamalla Charitha**

GitHub: https://github.com/charitha0310

---

*AgriSense AI – Making crop recommendations smarter through machine learning.*
