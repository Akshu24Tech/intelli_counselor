# 🎓 Intelli-Counselor: Your AI-Powered JoSAA Counselor

**Intelli-Counselor** is a smart web application that uses machine learning to demystify the complex JoSAA counseling process for JEE aspirants. It provides a data-driven, strategic, and personalized college preference list to help students make confident decisions.

---

**Problem Statement**
**The Solution**
**Key Features**
**Tech Stack**
**Project Structure**
**Setup & Installation**
**Usage**
**Future Scope**

---

## 🎯 Problem Statement

The JoSAA (Joint Seat Allocation Authority) counseling process is a high-stakes, stressful, and overwhelming task for millions of engineering aspirants in India. Students face several challenges:

* **Information Overload**: Sifting through thousands of combinations of colleges and branches across years of historical data is a monumental task.
* **Dynamic & Complex Data**: Cutoff ranks are not static; they fluctuate based on exam difficulty, candidate performance, and changes in seat matrixes. Manually identifying these trends is nearly impossible.
* **High-Stakes Decision**: A poorly constructed preference list can lead to a student missing out on a better college or, in the worst case, not being allotted any seat at all.

---

## 💡 The Solution

**Intelli-Counselor** transforms this chaotic process into a simple, data-driven, and intelligent one. It acts as a personal AI advisor by:

* **Analyzing Historical Data**: It processes years of official JoSAA counseling data to understand underlying trends.
* **Predicting Future Ranks**: It uses trained Machine Learning models (LightGBM) to forecast the opening and closing ranks for the upcoming academic year.
* **Generating Strategic Lists**: Based on the student's rank(s), it generates a personalized preference list, smartly categorized into Ambitious, Safe, and Backup options.

The application correctly mirrors the official JoSAA logic, using a student's JEE Advanced rank for IITs and their JEE Main rank for NITs, IIITs, and GFTIs, all within a single, unified interface.

---

## ✨ Key Features

* **Unified JoSAA Logic**: Accurately handles both JEE Main and JEE Advanced ranks in a single prediction pipeline, just like the real counseling process.
* **Dual Rank Prediction**: Predicts both the Opening and Closing Ranks to give students a clear "Safe Zone" for each branch.
* **Strategic Preference Lists**: Automatically categorizes results into Ambitious, Safe, and Backup choices to maximize admission chances.
* **Interactive Trend Charts**: Allows users to click on any result to instantly visualize the historical closing rank trend for that specific branch from 2016–2024, building trust and transparency.
* **Branch Comparison Tool**: Enables students to select up to 3 branches and view their predicted stats and historical trend charts side-by-side, helping resolve tough choices.
* **AI Explainability (Coming Soon)**: A planned feature to use SHAP (SHapley Additive exPlanations) to show the top factors influencing a particular prediction.

---

## 🛠️ Tech Stack

**Backend & ML**: Python, Pandas, NumPy, Scikit-learn, LightGBM, SHAP
**Frontend**: Streamlit
**Data Exploration**: Jupyter Notebooks, Matplotlib, Seaborn

---

## 📁 Project Structure

```
Intelli-Counselor/
|
|-- app/
|   `-- main.py             # The main Streamlit application script
|
|-- data/
|   |-- raw/                # Original yearly JoSAA CSV files
|   `-- processed/          # Combined and preprocessed data, and encoders
|
|-- models/
|   |-- closing_rank_model.joblib
|   |-- opening_rank_model.joblib
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_data_preprocessing.ipynb
|   `-- 03_model_development.ipynb
|
|-- README.md               # You are here!
`-- requirements.txt        # Project dependencies
```

---

## 🚀 Setup & Installation

**Clone the repository:**

```bash
git clone https://github.com/your-username/Intelli-Counselor.git
cd Intelli-Counselor
```

**Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**Install the dependencies:**

```bash
pip install -r requirements.txt
```

**Prepare the Data & Models:**

1. Place your raw historical JoSAA CSV files inside the `data/raw/` directory.
2. Run the Jupyter Notebooks in order (01 → 02 → 03) to process the data and train the models.
   This will populate the `data/processed/` and `models/` directories with the necessary files for the app to run.

---

## ▶️ Usage

Once the setup is complete, you can run the Streamlit application from the project's root directory:

```bash
streamlit run app/main.py
```

Your web browser should automatically open a new tab with the Intelli-Counselor application running locally.

---

## 🔮 Future Scope

* **Integrate AI Explainability**: Complete the SHAP integration to show users why the AI made a specific prediction.
* **Incorporate Institute Data**: Add more valuable data points to the results, such as NIRF rankings, average placement packages, and fees for each college.
* **User Accounts & Saved Lists**: Allow users to create accounts to save and manage their generated preference lists.
* **Deployment**: Deploy the application to a cloud service like Heroku or Streamlit Community Cloud to make it publicly accessible.

---