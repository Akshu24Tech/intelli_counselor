import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.ai_assistant as assistant

# Set the page configuration for a professional look
st.set_page_config(
    page_title="Intelli-Counselor",
    page_icon="🎓",
    layout="wide"
)

# --- Helper Function ---
# We need to define this function here so we can use it in our app
def get_institute_type(institute):
    """Categorizes institute into IIT, NIT, IIIT, or GFTI."""
    if 'Indian Institute of Technology' in str(institute): return 'IIT'
    if 'National Institute of Technology' in str(institute): return 'NIT'
    if 'Indian Institute of Information Technology' in str(institute): return 'IIIT'
    return 'GFTI'

# --- Load Assets ---
@st.cache_resource
def load_assets():
    """Loads the trained model, encoders, and necessary data."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'closing_rank_model.joblib')
    model = joblib.load(model_path)
    
    processed_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    encoders_path = os.path.join(processed_data_dir, 'encoders.joblib')
    encoders = joblib.load(encoders_path)
    
    try:
        assets = {
            "closing_model": joblib.load(os.path.join(models_dir, 'closing_rank_model.joblib')),
            "opening_model": joblib.load(os.path.join(models_dir, 'opening_rank_model.joblib')),
            "encoders": joblib.load(os.path.join(processed_data_dir, 'encoders.joblib')),
            "df_combined": pd.read_csv(os.path.join(processed_data_dir, 'combined_counseling_data.csv'), low_memory=False)
        }
        # Engineer the 'Institute_Type' column right after loading
        assets['df_combined']['Institute_Type'] = assets['df_combined']['Institute'].apply(get_institute_type)
        return assets
    except FileNotFoundError as e:
        st.error(f"Asset loading failed. A required file was not found: {e.filename}. Please ensure all model and data files are in the correct directories.")
        return None

# Load all assets
assets = load_assets()

# --- Initialize Session State ---
# This helps store results and user selections across interactions
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False
if 'compare_list' not in st.session_state:
    st.session_state.compare_list = []

# --- Main App UI ---
st.title("🎓 Intelli-Counselor: Your AI-Powered JoSAA Counselor")
st.info(
    """
    **How this tool works:** This predictor mirrors the official JoSAA process.
    - For **IITs**, your **JEE Advanced Rank** will be used for predictions.
    - For **NITs, IIITs, and GFTIs**, your **JEE Main Rank** will be used.
    """
)

if assets:
    # Unpack assets
    closing_model, opening_model, encoders, df_combined = assets['closing_model'], assets['opening_model'], assets['encoders'], assets['df_combined']
    mappings = {col: dict(zip(le.transform(le.classes_), le.classes_)) for col, le in encoders.items()}

    # --- User Input in Sidebar ---
    st.sidebar.header("Your Details")
    qualification_status = st.sidebar.radio("Which exam(s) have you qualified for?", ('JEE Mains & JEE Advanced', 'JEE Mains only'), key='qualification_status')
    
    student_adv_rank = None
    if qualification_status == 'JEE Mains & JEE Advanced':
        student_adv_rank = st.sidebar.number_input("Enter your JEE Advanced Rank:", min_value=1, max_value=40000, value=5000, key='adv_rank')
        student_main_rank = st.sidebar.number_input("Enter your JEE Main Rank (CRL):", min_value=1, max_value=1200000, value=15000, key='main_rank_adv')
    else:
        student_main_rank = st.sidebar.number_input("Enter your JEE Main Rank (CRL):", min_value=1, max_value=1200000, value=15000, key='main_rank_only')

    available_inst_types = list(df_combined['Institute_Type'].unique())
    if qualification_status == 'JEE Mains only' and 'IIT' in available_inst_types:
        available_inst_types.remove('IIT')
    
    selected_inst_types = st.sidebar.multiselect("Select Institute Types:", options=available_inst_types, default=available_inst_types)
    selected_quota = st.sidebar.selectbox("Select Your Quota:", options=list(mappings['Quota'].values()), index=list(mappings['Quota'].values()).index('AI'))
    
    generate_button = st.sidebar.button("✨ Generate My Preference List", use_container_width=True)

    if generate_button:
        st.session_state.compare_list = [] # Reset comparison on new generation
        
        st.session_state.student_adv_rank = student_adv_rank
        st.session_state.student_main_rank = student_main_rank
        
        with st.spinner("🧠 Analyzing possibilities with AI..."):
            unique_combinations = df_combined[df_combined['Year'] == 2024].drop_duplicates(subset=['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Institute_Type', 'Gender'])
            predict_df = unique_combinations.copy()
            predict_df['Year'] = 2025
            
            for col, le in encoders.items():
                predict_df[col + '_encoded'] = predict_df[col].apply(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            predict_df['Is_Female_Only'] = predict_df['Gender'].apply(lambda x: 1 if 'Female-only' in str(x) else 0)
            
            model_features = ['Year', 'Is_Female_Only', 'Institute_encoded', 'Academic Program Name_encoded', 'Quota_encoded', 'Seat Type_encoded', 'Institute_Type_encoded']
            X_predict = predict_df[model_features]
            
            predict_df['Predicted_Opening_Rank'] = np.round(opening_model.predict(X_predict)).astype(int)
            predict_df['Predicted_Closing_Rank'] = np.round(closing_model.predict(X_predict)).astype(int)
            predict_df['Predicted_Range'] = predict_df.apply(lambda row: f"{row['Predicted_Opening_Rank']} - {row['Predicted_Closing_Rank']}", axis=1)
            
            results_df = predict_df[predict_df['Institute_Type'].isin(selected_inst_types) & (predict_df['Quota'] == selected_quota)]

            def get_rank_for_comparison(row):
                if row['Institute_Type'] == 'IIT':
                    return student_adv_rank if student_adv_rank else np.inf
                else:
                    return student_main_rank
            results_df['student_rank_to_use'] = results_df.apply(get_rank_for_comparison, axis=1)

            st.session_state.ambitious_df = results_df[results_df['Predicted_Closing_Rank'] < results_df['student_rank_to_use']].sort_values('Predicted_Closing_Rank', ascending=False)
            st.session_state.safe_df = results_df[(results_df['Predicted_Opening_Rank'] <= results_df['student_rank_to_use']) & (results_df['Predicted_Closing_Rank'] >= results_df['student_rank_to_use'])].sort_values('Predicted_Closing_Rank')
            st.session_state.backup_df = results_df[results_df['Predicted_Opening_Rank'] > results_df['student_rank_to_use']].sort_values('Predicted_Opening_Rank')
            st.session_state.results_generated = True

    # --- Function to display results with checkboxes ---
    def display_results_with_checkboxes(dataframe, category_name):
        st.header(f"{category_name}")
        
        key_prefix = category_name.replace(' ', '_').lower()

        for index, row in dataframe.iterrows():
            cols = st.columns([1, 10, 3])
            unique_key = f"{key_prefix}_{index}"
            
            row_dict = row.to_dict()
            
            is_in_list = any(d['Institute'] == row_dict['Institute'] and d['Academic Program Name'] == row_dict['Academic Program Name'] for d in st.session_state.compare_list)
            
            is_checked = cols[0].checkbox("", value=is_in_list, key=unique_key, disabled=(len(st.session_state.compare_list) >= 3 and not is_in_list))
            
            cols[1].markdown(f"**{row['Institute']}**\n\n*{row['Academic Program Name']}*")
            cols[2].metric(label="Predicted Range", value=row['Predicted_Range'])

            if is_checked and not is_in_list:
                if len(st.session_state.compare_list) < 3:
                    st.session_state.compare_list.append(row_dict)
                    st.rerun()
            elif not is_checked and is_in_list:
                st.session_state.compare_list = [item for item in st.session_state.compare_list if not (item['Institute'] == row_dict['Institute'] and item['Academic Program Name'] == row_dict['Academic Program Name'])]
                st.rerun()

    # --- Display Results and Comparison Tool ---
    if st.session_state.results_generated:
        st.success("🎉 Your personalized JoSAA preference list is ready!")
        st.markdown("Select up to 3 choices from any category to compare.")
        
        display_results_with_checkboxes(st.session_state.ambitious_df.head(20), "✨ Ambitious Choices")
        display_results_with_checkboxes(st.session_state.safe_df.head(30), "✅ Safe Choices")
        display_results_with_checkboxes(st.session_state.backup_df.head(20), "🛡️ Backup Choices")

        # --- PART 5: BRANCH COMPARISON TOOL ---
        st.markdown("---")
        st.header("⚖️ Branch Comparison Tool")

        num_selected = len(st.session_state.compare_list)
        if num_selected > 0:
            st.write(f"You have selected **{num_selected}** option(s) for comparison.")
            
            if num_selected >= 2:
                if st.button(f"Compare {num_selected} Selected Branches", use_container_width=True):
                    compare_cols = st.columns(num_selected)
                    
                    for i, item in enumerate(st.session_state.compare_list):
                        with compare_cols[i]:
                            st.subheader(item['Institute'])
                            st.markdown(f"**{item['Academic Program Name']}**")
                            st.metric("Predicted 2025 Range", item['Predicted_Range'])
                            
                            history_df = df_combined[
                                (df_combined['Institute'] == item['Institute']) &
                                (df_combined['Academic Program Name'] == item['Academic Program Name']) &
                                (df_combined['Quota'] == item['Quota']) &
                                (df_combined['Seat Type'] == item['Seat Type']) &
                                (df_combined['Gender'] == item['Gender'])
                            ].sort_values('Year')
                            
                            final_round_history = history_df.loc[history_df.groupby('Year')['Round'].idxmax()]
                            
                            if not final_round_history.empty:
                                chart_data = final_round_history[['Year', 'Closing Rank']].set_index('Year')
                                st.line_chart(chart_data)
                            else:
                                st.warning("No historical data to plot.")
            else:
                st.warning("Please select at least 2 branches to enable comparison.")
        else:
            st.info("Use the checkboxes above to select up to 3 branches you want to compare.")

st.markdown("---")
st.header("🤖 Talk to Your AI Counselor")
st.markdown("Ask a question about your personalized results to get strategic advice.")

# We use the results stored in session state
if st.session_state.results_generated:
    user_question = st.text_input("e.g., 'What is a good strategy for my choice filling?' or 'Compare my top 2 safe options.'")

    if st.button("Ask the AI Assistant", use_container_width=True):
        if user_question:
            with st.spinner("🤖 Consulting the AI... please wait."):
                # Call the function from our new ai_assistant.py file
                ai_response = assistant.get_ai_insight(
                    question=user_question,
                    student_rank_adv=st.session_state.get('student_adv_rank'), # Get ranks from session state
                    student_rank_main=st.session_state.get('student_main_rank'),
                    ambitious_df=st.session_state.ambitious_df,
                    safe_df=st.session_state.safe_df,
                    backup_df=st.session_state.backup_df
                )
                st.markdown(ai_response)
        else:
            st.warning("Please ask a question.")
