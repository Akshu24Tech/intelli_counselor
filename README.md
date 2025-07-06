# 🎓 Intelli-Counselor: Your AI-Powered JoSAA Counselor

Intelli-Counselor is a smart web application that uses machine learning and a generative AI assistant to demystify the complex JoSAA counseling process for JEE aspirants. It provides a data-driven, strategic, and personalized college preference list to help students make confident decisions.

## Problem Statement

## The Solution

## Key Features

## Tech Stack

## Project Structure

## Setup & Installation

## Usage

## Future Scope

### 🎯 Problem Statement
The JoSAA (Joint Seat Allocation Authority) counseling process is a high-stakes, stressful, and overwhelming task for millions of engineering aspirants in India. Students face several challenges:

- **Information Overload**: Sifting through thousands of combinations of colleges and branches across years of historical data is a monumental task.
- **Dynamic & Complex Data**: Cutoff ranks are not static; they fluctuate based on exam difficulty, candidate performance, and changes in seat matrixes. Manually identifying these trends is nearly impossible.
- **High-Stakes Decision**: A poorly constructed preference list can lead to a student missing out on a better college or, in the worst case, not being allotted any seat at all.

---

### 💡 The Solution
Intelli-Counselor transforms this chaotic process into a simple, data-driven, and intelligent one. It acts as a personal AI advisor by:

- **Analyzing Historical Data**: It processes years of official JoSAA counseling data to understand underlying trends.
- **Predicting Future Ranks**: It uses trained Machine Learning models (LightGBM) to forecast the opening and closing ranks for the upcoming academic year.
- **Generating Strategic Lists**: Based on the student's rank(s), it generates a personalized preference list, smartly categorized into Ambitious, Safe, and Backup options.
- **Providing AI-Powered Insights**: It features an integrated AI assistant, powered by OpenAI's GPT models, that can answer strategic questions about the user's personalized results.

The application correctly mirrors the official JoSAA logic, using a student's JEE Advanced rank for IITs and their JEE Main rank for NITs, IIITs, and GFTIs, all within a single, unified interface.

---

### ✨ Key Features
- **Unified JoSAA Logic**: Accurately handles both JEE Main and JEE Advanced ranks in a single prediction pipeline.
- **AI Counseling Assistant**: An integrated chatbot powered by OpenAI that analyzes a student's personalized results to answer strategic questions like "What is a good choice-filling strategy for me?" or "Compare my top options for CSE."
- **Dual Rank Prediction**: Predicts both the Opening and Closing Ranks to give students a clear "Safe Zone" for each branch.
- **Strategic Preference Lists**: Automatically categorizes results into Ambitious, Safe, and Backup choices to maximize admission chances.
- **Interactive Trend Charts**: Allows users to click on any result to instantly visualize the historical closing rank trend for that specific branch.
- **Branch Comparison Tool**: Enables students to select up to 3 branches and view their predicted stats and historical trend charts side-by-side.

---

### 🛠️ Tech Stack
- **Backend & ML**: Python, Pandas, NumPy, Scikit-learn, LightGBM  
- **AI Assistant**: OpenAI API (GPT-3.5-Turbo)  
- **Frontend**: Streamlit  
- **Data Exploration**: Jupyter Notebooks, Matplotlib, Seaborn  

---

### 📁 Project Structure
```

Intelli-Counselor/
|
\|-- .streamlit/
\|   `-- secrets.toml        # Securely stores API keys
| |-- app/ |   `-- main.py             # The main Streamlit application script
|
\|-- data/
\|   |-- raw/                # Original yearly JoSAA CSV files
\|   `-- processed/          # Combined and preprocessed data, and encoders
| |-- models/ |   |-- final_rank_predictor_model.joblib |   `-- opening\_rank\_model.joblib
|
\|-- notebooks/
\|   |-- 01\_data\_exploration.ipynb
\|   |-- 02\_data\_preprocessing.ipynb
\|   `-- 03_model_development.ipynb
| |-- src/ |   `-- ai\_assistant.py     # Logic for the OpenAI assistant
|
\|-- README.md               # You are here!
\`-- requirements.txt        # Project dependencies

````

---

### 🚀 Setup & Installation

**Clone the repository:**
```bash
git clone https://github.com/your-username/Intelli-Counselor.git
cd Intelli-Counselor
````

**Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**Install the dependencies:**

```bash
pip install -r requirements.txt
```

**Set up OpenAI API Key:**

1. Get your API key from the OpenAI Platform.
2. Create a folder named `.streamlit` in the root project directory.
3. Inside `.streamlit`, create a file named `secrets.toml`.
4. Add your API key to the file like this:

```toml
OPENAI_API_KEY = "sk-YourSecretAPIKeyGoesHere"
```

**Prepare the Data & Models:**

* Place your raw historical JoSAA CSV files inside the `data/raw/` directory.
* Run the Jupyter Notebooks in order (`01 -> 02 -> 03`) to process the data and train the models.
  This will populate the `data/processed/` and `models/` directories with the necessary files for the app to run.

---

### ▶️ Usage

Once the setup is complete, you can run the Streamlit application from the project's root directory:

```bash
streamlit run app/main.py
```

Your web browser should automatically open a new tab with the Intelli-Counselor application running locally.

---

### 🔮 Future Scope

* **AI Explainability (SHAP)**: Integrate SHAP to show users why the base ML model made a specific rank prediction.
* **Incorporate Institute Data**: Add more valuable data points to the results, such as NIRF rankings, average placement packages, and fees for each college.
* **User Accounts & Saved Lists**: Allow users to create accounts to save and manage their generated preference lists.
* **Deployment**: Deploy the application to a cloud service like Heroku or Streamlit Community Cloud to make it publicly accessible.

