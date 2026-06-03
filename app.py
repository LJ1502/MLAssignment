import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import json

# --- 1. Load the Machine Learning Model ---
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\User\Documents\UM\Y1S2\WIA1006\GroupAssignment\dating_model.pkl")

model = load_model()

# --- 2. Setup the App & API Sidebar ---
st.title("🧞‍♂️ The Dating Profile Akinator")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key (Optional):", type="password")
    st.write("If left blank, the app will run in 'Offline Mode' and use mathematical averages for missing data.")

# Initialize the "Akinator" step counter
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.user_data = {}

st.progress(st.session_state.step / 4)

# --- STEP 1: Physical Demographics ---
if st.session_state.step == 1:
    st.subheader("Question 1: Let's start with the basics.")
    height = st.slider("What is your height (cm)?", 140, 220, 170)
    weight = st.slider("What is your weight (kg)?", 40, 150, 70)
    
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

# --- STEP 3: Smart Extraction ---
elif st.session_state.step == 3:
    st.subheader("Question 3: Show me what you're working with.")
    bio = st.text_area("Paste your bio here:")
    photos = st.file_uploader("Upload your profile photos (Select multiple):", 
                              type=['jpg', 'jpeg', 'png'], 
                              accept_multiple_files=True)
    
    if st.button("Consult the Akinator 🔮"):
        if bio and photos:
            
            # 1. HARDCODED METRICS (As requested)
            st.session_state.user_data['age'] = 24 
            st.session_state.user_data['profile_pics_count'] = len(photos)
            st.session_state.user_data['bio_length'] = len(bio)
            
            # 2. DYNAMIC LLM EXTRACTION (Text-only to save tokens)
            if api_key:
                with st.spinner("AI is reading your bio..."):
                    try:
                        genai.configure(api_key=api_key)
                        llm_model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        # We ask the LLM to count the hobbies in the text!
                        prompt = f"""
                        Analyze this dating profile bio and count how many distinct hobbies or interests are mentioned.
                        Return EXACTLY a JSON object with one key:
                        - 'num_interests': integer
                        Bio: "{bio}"
                        """
                        response = llm_model.generate_content(prompt)
                        clean_response = response.text.strip().replace('```json', '').replace('```', '')
                        ai_data = json.loads(clean_response)
                        
                        st.session_state.user_data['num_interests'] = ai_data.get('num_interests', 4)
                    except Exception as e:
                        # If you are out of tokens, it quietly falls back to 4 interests without crashing
                        st.warning("API Token Limit Reached. Switching to Offline Mode averages.")
                        st.session_state.user_data['num_interests'] = 4
            else:
                st.session_state.user_data['num_interests'] = 4 # Offline Default

            st.session_state.step += 1
            st.rerun()
        else:
            st.warning("Please provide a bio and upload at least one photo!")

# --- STEP 4: The Decision Tree Prediction ---
elif st.session_state.step == 4:
    st.subheader("The Akinator has made a decision...")
    
    with st.spinner("Aligning data and calculating popularity..."):
        
        expected_columns = model.feature_names_in_
        aligned_data = {col: 0 for col in expected_columns}
        
        # MEAN VALUES: Safe defaults for anything missing
        mean_values = {
            'swipe_right_ratio': 0.50,
            'num_interests': 4,
            'app_usage_time_min': 60,
            'age': 24,
            'bio_length': 100,
            'profile_pics_count': 3
        }
        
        for key, mean_val in mean_values.items():
            if key in aligned_data:
                aligned_data[key] = mean_val
        
        for key, value in st.session_state.user_data.items():
            if key in aligned_data:
                aligned_data[key] = value
                
        input_df = pd.DataFrame([aligned_data])
        prediction = model.predict(input_df)[0]
        
        # Display Results
        st.divider()
        if prediction == 'High':
            st.success("✨ THE AKINATOR SAYS: HIGH POPULARITY!")
            st.balloons()
        else:
            st.error("📉 THE AKINATOR SAYS: LOW POPULARITY.")
            
        st.write("### Data Used for Prediction:")
        st.json({
            "Age (Hardcoded)": st.session_state.user_data.get('age'),
            "Photos Uploaded": st.session_state.user_data.get('profile_pics_count'),
            "Bio Character Count": st.session_state.user_data.get('bio_length'),
            "Interests Count (LLM/Mean)": st.session_state.user_data.get('num_interests')
        })
        
        if st.button("Start Over 🔄"):
            st.session_state.step = 1
            st.session_state.user_data = {}
            st.rerun()