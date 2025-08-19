🩺 Heart Failure Prediction
โครงการนี้เป็นแอปพลิเคชัน Machine Learning ที่พัฒนาขึ้นเพื่อทำนายความเสี่ยงภาวะหัวใจล้มเหลว (Heart Failure) โดยใช้ข้อมูลผู้ป่วยทางคลินิก โครงการนี้มีจุดประสงค์เพื่อใช้เป็นเครื่องมือประเมินความเสี่ยงเบื้องต้นเพื่อช่วยในการตัดสินใจและให้ข้อมูลเชิงลึกแก่ผู้ใช้งาน

✨ คุณสมบัติหลัก
การทำนายความเสี่ยง: ใช้โมเดล Machine Learning เพื่อทำนายโอกาสเกิดภาวะหัวใจล้มเหลว

อินเทอร์เฟซที่ใช้งานง่าย: พัฒนาด้วย Streamlit ทำให้ผู้ใช้สามารถป้อนข้อมูลและดูผลลัพธ์ได้อย่างง่ายดาย

โมเดล Decision Tree: ใช้โมเดล Decision Tree Classifier ซึ่งเป็นโมเดลที่ตีความได้ง่ายและมีประสิทธิภาพ

💻 เทคโนโลยีที่ใช้
Python: ภาษาหลักในการพัฒนา

Pandas & NumPy: สำหรับการจัดการและวิเคราะห์ข้อมูล

Scikit-learn: สำหรับการสร้างและฝึกฝนโมเดล Machine Learning

Streamlit: สำหรับการสร้างเว็บแอปพลิเคชันเชิงโต้ตอบ

Joblib: สำหรับการบันทึกและโหลดโมเดลที่ฝึกฝนแล้ว

🚀 การติดตั้งและรันโปรเจกต์
ทำตามขั้นตอนเหล่านี้เพื่อรันโปรเจกต์บนเครื่องของคุณ:

Clone repository:

Bash

git clone https://github.com/IceEz555/Heart-Failure-Prediction.git
เข้าสู่ directory ของโปรเจกต์:

Bash

cd Heart-Failure-Prediction
สร้างและเปิดใช้งาน virtual environment (แนะนำ):

Bash

python -m venv venv
# สำหรับ Windows
venv\Scripts\activate
# สำหรับ macOS/Linux
source venv/bin/activate
ติดตั้ง dependencies ที่จำเป็น:

Bash

pip install -r requirements.txt
รันแอปพลิเคชัน Streamlit:

Bash

streamlit run app.py
หลังจากรันคำสั่งนี้ เว็บแอปพลิเคชันจะเปิดขึ้นในเบราว์เซอร์ของคุณโดยอัตโนมัติ

📊 ข้อมูลที่ใช้
โมเดลนี้ได้รับการฝึกฝนจากข้อมูลสาธารณะที่รวบรวมข้อมูลทางคลินิกที่สำคัญ เช่น อายุ, เพศ, ความดันโลหิต, ระดับคอเลสเตอรอล, และอื่นๆ เพื่อใช้ในการทำนายภาวะหัวใจล้มเหลว

🔗 ดูตัวอย่างการใช้งานจริง
หากโปรเจกต์นี้ได้รับการ Deploy บน Streamlit Cloud คุณสามารถเพิ่มลิงก์ที่นี่ได้ (ตัวอย่าง):
[🔗 Heart Failure Prediction (Live Demo)]([Your Streamlit App URL Here])

✍️ ผู้พัฒนา
IceEz555

📄 License
โปรเจกต์นี้อยู่ภายใต้ MIT License - โปรดดูไฟล์ LICENSE.md สำหรับรายละเอียดเพิ่มเติม (หากมี)
