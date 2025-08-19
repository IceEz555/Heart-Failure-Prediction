# Heart Failure Prediction

🚑 **Heart Failure Prediction** เป็นแอปพลิเคชัน Machine Learning ที่พัฒนาด้วย **Streamlit**  
ใช้สำหรับทำนายความเสี่ยงของภาวะหัวใจล้มเหลว โดยอ้างอิงข้อมูลคุณลักษณะทางคลินิก (Clinical Features)

---

## ✨ Features

- 🔍 **ทำนายความเสี่ยง (Risk Prediction):** ใช้โมเดล Decision Tree Classifier  
- 🖥️ **ใช้งานง่ายผ่านเว็บ (Web App):** ผู้ใช้สามารถกรอกข้อมูลสุขภาพแล้วรับผลลัพธ์ได้ทันที  
- ⚡ **โหลดโมเดลรวดเร็ว:** โมเดลถูกบันทึกด้วย **Joblib**  
- 📊 **โครงสร้างโปร่งใส:** ใช้ Decision Tree ที่เข้าใจได้ง่าย

---

## 🛠️ Tech Stack

- **Python** – ภาษาโปรแกรมหลัก  
- **Pandas, NumPy** – สำหรับประมวลผลข้อมูล  
- **Scikit-learn** – ใช้ในการสร้างและทดสอบโมเดล Machine Learning  
- **Streamlit** – สร้าง Web Application  
- **Joblib** – จัดเก็บและเรียกใช้งานโมเดลที่ฝึกแล้ว

---

## 📂 Model & Data

โมเดลนี้ถูกฝึกด้วยข้อมูลผู้ป่วยที่มีคุณลักษณะ (features) เช่น:

อายุ (Age)

เพศ (Sex)

ความดันโลหิต (Blood Pressure)

คอเลสเตอรอล (Cholesterol)

ค่าอื่น ๆ ตาม dataset

ผลลัพธ์ที่ได้:

High Risk → มีโอกาสสูงที่จะเกิดภาวะหัวใจล้มเหลว

Low Risk → มีโอกาสต่ำ

---

## 🌐 Demo

![Preview2](01_Preview/Predict.png)

👉 [Demo Online ](https://heart-failure-prediction-ai.streamlit.app/)


