import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Machine Learning Model ---
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\User\Documents\UM\Y1S2\WIA1006\GroupAssignment\dating_model_svm.pkl")
model = load_model()

# --- 2. Setup the App ---
st.title("🧞‍♂️ The Dating Profile Akinator")
st.write("Answer a few questions and let the AI predict your profile popularity!")

# Initialize the "Akinator" step counter
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.user_data = {}

st.progress(st.session_state.step / 4)

# --- STEP 1: Physical Demographics ---
if st.session_state.step == 1:
    st.subheader("Question 1: Let's start with the basics.")
    
    height = st.number_input("What is your height (cm)?", min_value=100, max_value=250, value=170)
    weight = st.number_input("What is your weight (kg)?", min_value=30, max_value=200, value=70)
    
    if st.button("Next ➡️"):
        st.session_state.user_data['height'] = height
        st.session_state.user_data['weight'] = weight
        st.session_state.step += 1
        st.rerun()

# --- STEP 2: Behavioral Data ---
elif st.session_state.step == 2:
    st.subheader("Question 2: How addicted are you to the app?")
    usage = st.slider("Daily App Usage (Minutes)", 0, 300, 60)
    
    if st.button("Next ➡️"):
        st.session_state.user_data['app_usage_time_min'] = usage
        st.session_state.step += 1
        st.rerun()

# --- STEP 3: Content Extraction (Offline) ---
elif st.session_state.step == 3:
    st.subheader("Question 3: Show me what you're working with.")
    bio = st.text_area("Paste your bio here:")
    photos = st.file_uploader("Upload your profile photos (Select multiple):", 
                              type=['jpg', 'jpeg', 'png'], 
                              accept_multiple_files=True)
    
    if st.button("Consult the Akinator 🔮"):
        if bio and photos:
            # 1. User Inputted Metrics
            st.session_state.user_data['profile_pics_count'] = len(photos)
            st.session_state.user_data['bio_length'] = len(bio)
            
            # 2. Hardcoded "True Neutral" Means for missing data
            st.session_state.user_data['age'] = 27 
            st.session_state.user_data['num_interests'] = 4
            
            st.session_state.step += 1
            st.rerun()
        else:
            st.warning("Please provide a bio and upload at least one photo!")

# --- STEP 4: The Decision Tree / SVM Prediction ---
elif st.session_state.step == 4:
    st.subheader("The Akinator has made a decision...")
    
    with st.spinner("Aligning data and calculating popularity..."):
        
        expected_columns = model.feature_names_in_
        aligned_data = {col: 0 for col in expected_columns}
        
        # MEAN VALUES: The "True Neutral" profile to balance the SVM
        mean_values = {
            'swipe_right_ratio': 0.51,
            'num_interests': 4,
            'app_usage_time_min': 52,
            'age': 27,
            'bio_length': 111,
            'profile_pics_count': 3
        }
        
        for key, mean_val in mean_values.items():
            if key in aligned_data:
                aligned_data[key] = mean_val
        
        for key, value in st.session_state.user_data.items():
            if key in aligned_data:
                aligned_data[key] = value
                
        input_df = pd.DataFrame([aligned_data])
        
        # ==========================================================
        # 🐛 TERMINAL DEBUGGING LOGS (Your requested modification!)
        # ==========================================================
        print("\n" + "="*50)
        print("🚀 RUNNING NEW PREDICTION...")
        print("EXACT DATA SENT TO THE SVM MODEL:")
        print("-" * 50)
        # This pandas trick prints every single column vertically without hiding anything
        print(input_df.iloc[0].to_string()) 
        print("="*50 + "\n")
        # ==========================================================

        prediction = model.predict(input_df)[0]
        
        # Display Results
        st.divider()
        if prediction == 'High':
            st.success("✨ THE AKINATOR SAYS: HIGH POPULARITY!")
            st.balloons()
        else:
            st.error("📉 THE AKINATOR SAYS: LOW POPULARITY.")
        
        if st.button("Start Over 🔄"):
            st.session_state.step = 1
            st.session_state.user_data = {}
            st.rerun()