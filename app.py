import joblib
import pandas as pd
import streamlit as st

# โหลดโมเดล Decision Tree
model_DecisionTree = joblib.load(r"C:\Users\LENOVO\Desktop\Project_Ai\DecisionTree_model.joblib")

# ตั้งค่าหน้าหลัก
st.set_page_config(page_title="Heart Failure Prediction", page_icon="❤️")

# สไตล์ Minimal (CSS)
st.markdown(
    """
    <style>
    body { background-color: #f8f9fa; color: #333333; font-family: Arial, sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; padding: 12px; font-size: 16px; background-color: #ff4b4b; color: white; }
    .stButton>button:hover { background-color: #ff3333; }
    .stTitle { text-align: center; }
    .stSidebar { background-color: #f1f1f1; padding: 20px; }
    .prediction-result { text-align: center; font-size: 18px; padding: 10px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# หัวข้อ
st.title("❤️ Heart Failure Prediction")

# รายการค่าที่ผู้ใช้ต้องเลือก
sex_list_selection = ["ชาย", "หญิง"]
ChestPainType_list_selection = ["TA: เจ็บหน้าอกแบบโรคหลอดเลือดหัวใจ", "ATA: เจ็บหน้าอกแบบผิดปกติ", "NAP: เจ็บหน้าอกที่ไม่เกี่ยวกับโรคหัวใจ", "ASY: ไม่มีอาการเจ็บหน้าอก"]
FastingBS_list_selection = ["Yes", "No"]
RestingECG_list_selection = ["Normal", "ST: คลื่นไฟฟ้าหัวใจผิดปกติ", "LVH: มีภาวะกล้ามเนื้อหัวใจหนาตัว"]
ExerciseAngina_list_selection = ["Yes", "No"]
ST_Slope_list_selection = ["Up: สูงขึ้น", "Flat: คงที่", "Down: ลดลง"]

# ฟอร์มป้อนข้อมูล
with st.form("prediction_form"):
    st.markdown("### 🔍 Patient Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("อายุ", 0, 100, 35)
        sex = st.radio("เพศ", sex_list_selection)
        ChestPainType = st.radio("อาการเจ็บหน้าอก", ChestPainType_list_selection)
        FastingBS = st.selectbox("ค่าน้ำตาลในเลือดขณะอดอาหาร (>120 mg/dl)", FastingBS_list_selection)
        RestingECG = st.radio("คลื่นไฟฟ้าหัวใจขณะพัก", RestingECG_list_selection)

    with col2:
        RestingBP = st.slider("ค่าความดันโลหิตขณะพัก (mmHg)", 40, 200, 120)
        Cholesterol = st.slider("ค่าคอเลสเตอรอลในเลือด (mg/dL)", 50, 400, 200)
        MaxHR = st.slider("อัตราการเต้นของหัวใจสูงสุด (bpm)", 50, 250, 120)
        Oldpeak = st.slider("ประเมินภาวะขาดเลือด (ค่า ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        ST_Slope = st.selectbox("ลักษณะความชันของ ST segment", ST_Slope_list_selection)
    
    ExerciseAngina = st.selectbox("อาการเจ็บหน้าอกขณะออกกำลังกาย", ExerciseAngina_list_selection)

    # ปุ่มพยากรณ์
    submit_button = st.form_submit_button("🔍 Predict")

# แปลงค่าหมวดหมู่ให้เป็นตัวเลข
convert_sex = {"ชาย": 1, "หญิง": 0}
convert_ChestPainType = {"TA: เจ็บหน้าอกแบบโรคหลอดเลือดหัวใจ": 3, "ATA: เจ็บหน้าอกแบบผิดปกติ": 1, "NAP: เจ็บหน้าอกที่ไม่เกี่ยวกับโรคหัวใจ": 2, "ASY: ไม่มีอาการเจ็บหน้าอก": 0}
convert_FastingBS = {"Yes": 1, "No": 0}
convert_RestingECG = {"Normal": 1, "ST: คลื่นไฟฟ้าหัวใจผิดปกติ": 2, "LVH: มีภาวะกล้ามเนื้อหัวใจหนาตัว": 0}
convert_ExerciseAngina = {"Yes": 1, "No": 0}
convert_ST_Slope = {"Up: สูงขึ้น": 2, "Flat: คงที่": 1, "Down: ลดลง": 0}

if submit_button:
    try:
        df = pd.DataFrame(
            {
                "Age": [age],
                "Sex": [convert_sex[sex]],
                "ChestPainType": [convert_ChestPainType[ChestPainType]],
                "RestingBP": [RestingBP],
                "Cholesterol": [Cholesterol],
                "FastingBS": [convert_FastingBS[FastingBS]],
                "RestingECG": [convert_RestingECG[RestingECG]],
                "MaxHR": [MaxHR],
                "ExerciseAngina": [convert_ExerciseAngina[ExerciseAngina]],
                "Oldpeak": [Oldpeak],
                "ST_Slope": [convert_ST_Slope[ST_Slope]],
            },
            index=[0],
        )

        # ทำการพยากรณ์
        prediction = model_DecisionTree.predict(df)[0]

        # แสดงผลลัพธ์
        st.markdown("### 📝 Prediction Result")
        if prediction == 1:
            st.markdown('<div class="prediction-result" style="background-color: #ffdddd; color: red;">⚠️ High Risk of Heart Disease</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-result" style="background-color: #d4edda; color: green;">✅ Low Risk of Heart Disease</div>', unsafe_allow_html=True)

    except ValueError:
        st.error("⚠️ Please ensure all fields are correctly filled!")
