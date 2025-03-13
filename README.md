# 🚀 PocketPilot - Your Smart Fare Predictor!  

<p align="center">
  <img src="https://github.com/user-attachments/assets/3a338b5a-7ace-4032-bf7c-e84fcf34280e" alt="Moving Car" width="500"/>
</p>

## 🎯 What is PocketPilot?  
PocketPilot is your **personal travel cost guru!** 🚖✨ This **Django-powered machine learning app** predicts car fares with **laser-sharp accuracy**, using a **Random Forest model**. Say goodbye to fare surprises! 🎩✨

## 🌟 Why You'll Love It  
- 🎯 **ML-Powered Predictions** – Get instant and precise fare estimates!
- 🎨 **Sleek & Simple UI** – No rocket science, just enter details & get your fare!
- 🔄 **Preprocessing Magic** – Clean data, smart analysis, better results!
- 🌎 **API-Ready** – Easily extendable for integrations!

## 🛠 Tech Ingredients  
- **Backend:** Django Framework 🍃  
- **ML Goodies:** Scikit-Learn, Pandas, NumPy 🤖  
- **Frontend:** HTML, CSS, Bootstrap 🎨  
- **Deployment Ready?** Yes! AWS/GCP compatible 🚀  

## 🗂 Project Roadmap  
```
├─ ML_Deployment
│  ├─ manage.py
│  ├─ ML_Deployment
│  │  ├─ asgi.py
│  │  ├─ settings.py
│  │  ├─ urls.py
│  │  ├─ wsgi.py
│  └─ PocketPilot
│     ├─ admin.py
│     ├─ apps.py
│     ├─ ml_models
│     │  ├─ preprocessor.pkl
│     │  └─ random_forest_model.pkl
│     ├─ models.py
│     ├─ templates
│     │  ├─ home.html
│     │  ├─ predict.html
│     │  └─ result.html
│     ├─ tests.py
│     ├─ urls.py
│     ├─ views.py
├─ Preprocessing_And_Model_Training.ipynb
└─ requirements.txt
```

## 🚀 Let's Get You Started!  
### **1️⃣ Clone the Magic!**  
```bash
git clone https://github.com/yourusername/PocketPilot.git
cd PocketPilot
```
### **2️⃣ Set Up a Safe Zone (Virtual Env)**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### **3️⃣ Load Up the Goodies!**  
```bash
pip install -r requirements.txt
```
### **4️⃣ Fire Up the Engine!**  
```bash
python manage.py runserver
```
🌍 **Visit:** http://127.0.0.1:8000/ and let the fare magic begin! 🎩✨

## 🧠 Model Training Breakdown  
- All the ML wizardry happens in **Preprocessing_And_Model_Training.ipynb** 🧪🔬
- The trained model (`random_forest_model.pkl`) and **preprocessing pipeline** (`preprocessor.pkl`) work their charm to predict fares. 🏎💨

## 🏆 How to Use PocketPilot?  
1️⃣ Open the web app.  
2️⃣ Enter trip details (distance, time, passengers, etc.).  
3️⃣ Click **Predict Fare** and watch the magic unfold! ✨

## 🎯 Future Enhancements  
- 🚀 **Deploy on AWS/GCP** – Reach the masses!
- 📈 **Boost Model Accuracy** – More data, smarter predictions!
- 📱 **Mobile-Friendly UI** – Predict fares on the go!

## 📜 License  
MIT License – **Because sharing is caring!** ❤️

## 🤝 Join the Fun!  
Want to improve PocketPilot? Fork, tweak, and submit a PR! 🚀🔥

## 👩‍💻 Created by: **Habiba Mowafy** 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/habiba-el-sayed)

---
🌟 **Love PocketPilot? Give it a ⭐ on GitHub!** 🚖✨
