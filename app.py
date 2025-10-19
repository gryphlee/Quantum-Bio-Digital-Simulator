import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import time
import base64
from pathlib import Path
from datetime import datetime
import re # Using re for the local extractor

# --- HELPER FUNCTIONS ---

@st.cache_data
def get_img_as_base64(file):
    if not Path(file).is_file(): return None
    with open(file, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.insert(0, f"[{timestamp}] {message}")
    if len(st.session_state.log) > 15: st.session_state.log.pop()

# --- LOCAL OFFLINE EXTRACTOR ---
def extract_vitals_locally_with_regex(notes):
    """Extracts vitals using regular expressions. No API or internet needed."""
    vitals = {}
    notes_lower = notes.lower()
    patterns = {
        'glucose': r'glucose.*?(\d+\.?\d*)',
        'heart_rate': r'(?:heart rate|hr|bpm)[\s:]*?(\d+)',
        'bp_systolic': r'(?:bp|blood pressure)[\s:]*?(\d+)\s*/',
        'bp_diastolic': r'(?:bp|blood pressure)[\s:]*?\d+\s*/\s*(\d+)',
        'respiration': r'(?:respiration|resp)[\s:]*?(\d+)',
        'temperature': r'(?:temperature|temp)[\s:]*?(\d+\.?\d*)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, notes_lower)
        vitals[key] = float(match.group(1)) if match else None
    return vitals

# --- SIMULATION CORE LOGIC ---
def calculate_initial_vitals(profile):
    base_hr = 75-(profile['age']-45)//3+(5 if profile['gender']=='Female' else 0)+int(profile['activity_level']*5)
    base_bp_sys = 120+(profile['weight']-70)//2
    base_bp_dia = 80+(profile['weight']-70)//4
    return {'glucose':[profile['baseline_glucose']],'heart_rate':[base_hr],'bp_systolic':[base_bp_sys],'bp_diastolic':[base_bp_dia],'respiration':[16],'temperature':[37.0]}
def update_all_vitals(profile, compound_effects):
    last_vitals={k:v[-1] for k,v in st.session_state.vitals.items()}
    input_df=pd.DataFrame([{'current_glucose':last_vitals['glucose'],'insulin_dose':compound_effects['insulin'],'carb_intake':compound_effects['glucose']}])
    predicted_glucose=model.predict(input_df)[0]
    st.session_state.vitals['glucose'].append(round(predicted_glucose,2))
    glucose_delta=predicted_glucose-profile['baseline_glucose']
    new_hr=max(50,min(180,last_vitals['heart_rate']+(glucose_delta/20)+compound_effects['stimulant']-(profile['activity_level']*0.1)))
    st.session_state.vitals['heart_rate'].append(round(new_hr))
    new_bp_sys=last_vitals['bp_systolic']+(glucose_delta/15)+(compound_effects['stimulant']*2)
    new_bp_dia=last_vitals['bp_diastolic']+(glucose_delta/25)+compound_effects['stimulant']
    st.session_state.vitals['bp_systolic'].append(round(new_bp_sys));st.session_state.vitals['bp_diastolic'].append(round(new_bp_dia))
    new_resp=max(10,min(30,16+(abs(glucose_delta)/25)+(compound_effects['stimulant']/2)-compound_effects['sedative']))
    st.session_state.vitals['respiration'].append(round(new_resp))
    new_temp=max(36.0,min(40.0,last_vitals['temperature']+(compound_effects['stimulant']*0.05)-(compound_effects['sedative']*0.05)))
    st.session_state.vitals['temperature'].append(round(new_temp,1))
    for key in st.session_state.vitals:
        if len(st.session_state.vitals[key])>50:st.session_state.vitals[key].pop(0)

# --- CHECK FOR QUANTUM MODULE ---
try:
    from quantum_module import run_quantum_simulation
    quantum_enabled = True
except (ImportError, ModuleNotFoundError):
    quantum_enabled = False

# --- PAGE CONFIG & CSS ---
st.set_page_config(page_title="Universal In Silico Lab", page_icon="🔬", layout="wide")
st.markdown("""<style>.hologram-image-container{display:flex;justify-content:center;align-items:center;min-height:300px}.hologram-image{max-width:80%;filter:drop-shadow(0 0 1.2rem rgba(0,176,240,.7));animation:pulse 4s infinite ease-in-out}.hologram-image-stressed{max-width:80%;filter:drop-shadow(0 0 1.5rem rgba(255,65,54,.8));animation:pulse-stressed 2s infinite ease-in-out}@keyframes pulse{0%{transform:scale(1)}50%{transform:scale(1.02)}100%{transform:scale(1)}}@keyframes pulse-stressed{0%{transform:scale(1)}50%{transform:scale(1.05)}100%{transform:scale(1)}}</style>""", unsafe_allow_html=True)

# --- SESSION STATE INIT ---
if 'log' not in st.session_state: st.session_state.log = ["Lab Initialized. Configure patient or input data."]
if 'vitals' not in st.session_state: st.session_state.vitals = {'glucose': [90], 'heart_rate': [75], 'bp_systolic': [120], 'bp_diastolic': [80], 'respiration': [16], 'temperature': [37.0]}
if 'patient_db' not in st.session_state: st.session_state.patient_db = {}

# --- MODEL LOADING ---
model = pickle.load(open("patient_model.pkl", "rb")) if Path("patient_model.pkl").is_file() else None

# --- SIDEBAR ---
st.sidebar.title("🔬 Experiment Controls")
st.sidebar.markdown("---")

with st.sidebar.expander("📊 Patient Data Hub", expanded=True):
    patient_id = st.text_input("Patient ID", "P1001")
    
    # --- MERGED AND FLEXIBLE DATA INPUT ---
    data_input_tab1, data_input_tab2 = st.tabs(["File Upload", "Load Patient"])

    with data_input_tab1:
        st.write("**Upload Patient File**")
        uploaded_file = st.file_uploader("Upload a patient's notes (.txt) or vitals (.csv) file.", type=['txt', 'csv'])
        
        if st.button("Process File"):
            if uploaded_file is not None:
                # Logic to handle different file types
                if uploaded_file.name.endswith('.txt'):
                    with st.spinner("Processing notes locally..."):
                        content = uploaded_file.read().decode("utf-8")
                        extracted_vitals = extract_vitals_locally_with_regex(content)
                        st.session_state.patient_db[patient_id] = extracted_vitals
                        st.success(f"Notes processed for {patient_id}!")
                        add_log(f"Extracted vitals from {uploaded_file.name}.")
                        st.json(extracted_vitals)

                elif uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file)
                        df.columns = df.columns.str.lower()
                        latest_vitals = df.iloc[-1].to_dict()
                        st.session_state.patient_db[patient_id] = latest_vitals
                        st.success(f"CSV data processed for {patient_id}!")
                        add_log(f"Loaded vitals from {uploaded_file.name}.")
                        st.dataframe(df.tail(1))
                    except Exception as e:
                        st.error(f"Failed to process CSV file: {e}")
            else:
                st.warning("Please upload a file first.")
    
    with data_input_tab2:
        st.write("**Load Patient Data**")
        if not st.session_state.patient_db:
            st.caption("No patient data has been processed in this session.")
        else:
            patient_to_load = st.selectbox("Select Patient to Load", options=list(st.session_state.patient_db.keys()))
            if st.button("Load Patient Vitals"):
                loaded_vitals = st.session_state.patient_db[patient_to_load]
                for key, value in loaded_vitals.items():
                    if pd.notna(value) and key in st.session_state.vitals:
                        st.session_state.vitals[key] = [value]
                add_log(f"Vitals loaded for patient {patient_to_load}.")
                st.success(f"Vitals for {patient_to_load} are now active in the simulator!")

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Simulation Configuration"):
    patient_profile = {}
    patient_profile['age'] = st.slider("Age", 20, 80, 45)
    patient_profile['weight'] = st.slider("Weight (kg)", 40, 120, 70)
    patient_profile['height'] = st.slider("Height (cm)", 140, 200, 170)
    patient_profile['gender'] = st.selectbox("Gender", ["Male", "Female"])
    patient_profile['activity_level'] = st.select_slider("Activity Level", options=["Sedentary", "Light", "Moderate", "Active"], value="Light")
    activity_map = {"Sedentary": 0, "Light": 1, "Moderate": 2, "Active": 3}
    patient_profile['activity_level'] = activity_map[patient_profile['activity_level']]
    patient_profile['baseline_glucose'] = st.number_input("Baseline Glucose (mg/dL)", 70, 180, 90)
    if st.button("Set Patient Baseline"):
        st.session_state.vitals = calculate_initial_vitals(patient_profile)
        add_log(f"New patient baseline set.")
    st.markdown("---")
    compound_name = st.text_input("Compound Name", "Universal Drug")
    compound_effects = {}
    compound_effects['glucose'] = st.slider("Glucose Potency (Carb Eq.)", -20, 20, 0, key="glucose_slider")
    compound_effects['insulin'] = st.slider("Insulin Efficacy (Insulin Eq.)", -20, 20, 0, key="insulin_slider")
    compound_effects['stimulant'] = st.slider("Stimulant Effect (HR/BP)", -5, 5, 0, key="stimulant_slider")
    compound_effects['sedative'] = st.slider("Sedative Effect (Resp)", -2, 2, 0, key="sedative_slider")
    if st.button("Run Universal AI Experiment"):
        if model:
            add_log(f"Administering '{compound_name}'.")
            update_all_vitals(patient_profile, compound_effects)
        else:
            st.sidebar.error("AI model not found!")
st.sidebar.markdown("---")
with st.sidebar.expander("⚛️ Quantum Lab"):
    if not quantum_enabled: st.warning("Quantum Module Status: **OFFLINE**.")
    else:
        st.success("Quantum Module Status: **ONLINE**")
        molecule_to_sim = st.selectbox("Select Molecule:", ["LiH", "H2"])
        if st.button(f"Run Quantum Sim for {molecule_to_sim}"):
            with st.spinner(f"Running Quantum Simulation..."):
                energy = run_quantum_simulation(molecule_to_sim)
            st.session_state.last_quantum_energy = energy
            st.session_state.last_molecule_simulated = molecule_to_sim
            add_log(f"Quantum sim complete for {molecule_to_sim}. Energy: {energy:.4f} H.")
    if 'last_quantum_energy' in st.session_state:
        st.metric(f"Energy of {st.session_state.last_molecule_simulated} (H)", f"{st.session_state.last_quantum_energy:.6f}")

# --- MAIN DASHBOARD ---
st.title("🔬 Universal In Silico Laboratory")
st.caption(f"A configurable digital twin platform for predictive medicine.")
last_vitals = {key: val[-1] for key, val in st.session_state.vitals.items()}
col1, col2, col3 = st.columns((1.2, 1.2, 1))
with col1:
    st.header("❤️ Patient Vitals (AI-Powered)")
    vitals_tab1, vitals_tab2, vitals_tab3 = st.tabs(["Key Vitals", "Blood Pressure", "Other Metrics"])
    with vitals_tab1:
        df_key = pd.DataFrame({'Glucose (mg/dL)': st.session_state.vitals['glucose'], 'Heart Rate (bpm)': st.session_state.vitals['heart_rate']})
        st.line_chart(df_key)
    with vitals_tab2:
        df_bp = pd.DataFrame({'Systolic (mmHg)': st.session_state.vitals['bp_systolic'], 'Diastolic (mmHg)': st.session_state.vitals['bp_diastolic']})
        st.line_chart(df_bp)
    with vitals_tab3:
        df_other = pd.DataFrame({'Respiration (br/min)': st.session_state.vitals['respiration'], 'Temperature (°C)': st.session_state.vitals['temperature']})
        st.area_chart(df_other)
    st.header("📋 Simulation Log")
    log_container = st.container(height=150)
    for entry in st.session_state.log: log_container.text(entry)
with col2:
    st.header("🧬 System View (Patient Twin)")
    view_options = ["Full Body", "Holographic Overview", "Brain", "Heart", "Lungs", "Liver", "Stomach", "Kidneys"]
    selected_view = st.radio("Select View Focus:", view_options, horizontal=True, label_visibility="collapsed")
    if selected_view == "Holographic Overview":
        organ_names, pos_x, pos_y, base_sizes = ['Brain', 'Heart', 'Lungs (L)', 'Lungs (R)', 'Liver', 'Stomach', 'Kidneys'], [2.5, 2.5, 1.5, 3.5, 3, 2, 2.5], [4, 3, 2.8, 2.8, 1, 1, 0], [40, 50, 30, 30, 35, 35, 30]
        sizes, colors, glow_colors, edge_color = list(base_sizes), ['#00B0F0']*7, ['rgba(0,176,240,0.3)']*7, '#3D3D3D'
        if last_vitals['glucose'] > 150:
            edge_color='#FF4136'; colors[0],glow_colors[0],sizes[0]='#FFA500','rgba(255,165,0,0.5)',base_sizes[0]*1.1; colors[1],glow_colors[1],sizes[1]='#FF4136','rgba(255,65,54,0.7)',base_sizes[1]*1.2; colors[6],glow_colors[6],sizes[6]='#FF851B','rgba(255,133,27,0.5)',base_sizes[6]*1.1
        elif last_vitals['glucose'] < 70:
            edge_color='#B10DC9'; colors[0],glow_colors[0],sizes[0]='#AAAAAA','rgba(170,170,170,0.4)',base_sizes[0]*0.9; colors[1],glow_colors[1],sizes[1]='#B10DC9','rgba(177,13,201,0.5)',base_sizes[1]*1.1
        edge_x,edge_y = [],[]; connections=[(0,1),(1,2),(1,3),(1,4),(1,5),(4,6),(5,6)]; [edge_x.extend([pos_x[s],pos_x[e],None]) or edge_y.extend([pos_y[s],pos_y[e],None]) for s,e in connections]
        fig=go.Figure(); fig.add_trace(go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(width=2.5,color=edge_color),hoverinfo='none')); fig.add_trace(go.Scatter(x=pos_x,y=pos_y,mode='markers',marker=dict(size=[s*1.5 for s in sizes],color=glow_colors),hoverinfo='none')); fig.add_trace(go.Scatter(x=pos_x,y=pos_y,mode='markers+text',marker=dict(size=sizes,color=colors,line=dict(width=2,color='rgba(255,255,255,0.8)')),text=organ_names,textposition="middle center",textfont=dict(color='white',size=10))); fig.update_layout(xaxis=dict(visible=False),yaxis=dict(visible=False),showlegend=False,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=350,margin=dict(l=0,r=0,t=0,b=0)); st.plotly_chart(fig,use_container_width=True)
    else:
        organ_data = {"Full Body": {"path": "assets/full_body.png", "normal": "All Systems Nominal", "high": "System-Wide Stress", "low": "System-Wide Fatigue"},"Brain":     {"path": "assets/brain.png",   "normal": "Normal", "high": "Cognitive Fog", "low": "Low Energy"},"Heart":     {"path": "assets/heart.png",   "normal": "Stable", "high": "High Stress", "low": "Arrhythmia Risk"},"Lungs":     {"path": "assets/lungs.png",   "normal": "Clear", "high": "Shallow Breathing", "low": "Normal"},"Liver":     {"path": "assets/liver.png",   "normal": "Normal", "high": "Processing Overload", "low": "Normal"},"Stomach":   {"path": "assets/stomach.png", "normal": "Normal", "high": "Digestive Stress", "low": "Normal"},"Kidneys":   {"path": "assets/kidneys.png", "normal": "Normal", "high": "High Filtration Load", "low": "Normal"},}
        data = organ_data[selected_view]
        status, css_class = data["normal"], "hologram-image"
        if last_vitals['glucose'] > 150: status, css_class = data["high"], "hologram-image-stressed"
        elif last_vitals['glucose'] < 70: status = data["low"]
        st.markdown(f"**Focus: {selected_view} Status**")
        img_col, data_col = st.columns([0.6, 0.4])
        with img_col:
            img = get_img_as_base64(data["path"])
            if img: st.markdown(f'<div class="hologram-image-container"><img src="data:image/png;base64,{img}" class="{css_class}"></div>', unsafe_allow_html=True)
            else: st.error(f"Image not found: {data['path']}.")
        with data_col:
            st.metric("Current Status", status)
            st.metric("Heart Rate", f"{last_vitals['heart_rate']} bpm")
            st.metric("Blood Pressure", f"{last_vitals['bp_systolic']}/{last_vitals['bp_diastolic']} mmHg")
            st.metric("Respiration", f"{last_vitals['respiration']} br/min")
            st.metric("Temperature", f"{last_vitals['temperature']} °C")
with col3:
    st.header("⚛️ Quantum Analysis")
    st.markdown("Molecular properties of hypothetical drugs.")
    if 'last_quantum_energy' in st.session_state and 'last_quantum_energy' in st.session_state:
        energy,mol_name = st.session_state.last_quantum_energy, st.session_state.last_molecule_simulated
        fig_q=go.Figure(go.Scatter(x=[0,1,1,0,-1,-1,0],y=[1,0.5,-0.5,-1,-0.5,0.5,1],mode='lines',fill="toself",fillcolor='rgba(0,176,240,0.3)',line=dict(color='rgba(0,176,240,1)',width=3)))
        fig_q.update_layout(title=f"Simulated Molecule: {mol_name}",xaxis=dict(visible=False,range=[-2,2]),yaxis=dict(visible=False,range=[-1.5,1.5]),height=250, margin=dict(l=0,r=0,t=40,b=0),plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_q, use_container_width=True)
        st.metric(label=f"Simulated Energy (Hartree)", value=f"{energy:.6f}")
        st.caption("Lower energy often indicates a more stable molecular structure.")
    else:
        st.info("Run a Quantum Simulation from the sidebar to display molecular analysis here.")
st.markdown("---")
st.markdown("Project 'In Silico' Final Professional Edition")









