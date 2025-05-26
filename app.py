
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="VP-AI", layout="wide")
st.title("🧬 AI Dự đoán Tác nhân và Gợi ý Kháng sinh")

@st.cache_data
def load_and_train():
    df = pd.read_csv("Mô hình AI.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={"Benh ngay thu truoc khi nhap vien": "Benh ngay thu", " SpO2": "SpO2"}, inplace=True)
    df["Tuoi"] = df["Tuoi"].apply(lambda x: float(str(x).replace("Thg", "").strip()) / 10 if "thg" in str(x).lower() else pd.to_numeric(x, errors="coerce"))
    df["Benh ngay thu"] = pd.to_numeric(df["Benh ngay thu"], errors="coerce")
    df["SpO2"] = pd.to_numeric(df["SpO2"], errors="coerce")

    # Nhị phân hóa các cột object còn lại
    text_cols = df.select_dtypes(include="object").columns.difference(["Tac nhan", "ID", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện"])
    df[text_cols] = df[text_cols].applymap(lambda x: 1 if str(x).strip().lower() in ["x", "có", "yes"] else (0 if str(x).strip().lower() in ["không", "khong", "/", "no"] else np.nan))

    df = df[df["Tac nhan"].notna()]
    X = df.drop(columns=["Tac nhan", "ID", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện",
                         "So ngay dieu tri",
                         "Erythomycin", "Tetracyline", "Chloranphenicol", "Arithromycin", "Ampicillin",
                         "Ampicillin-Sulbalactam", "Cefuroxime", "Cefuroxime Axetil", "Ceftazidine", "Viprofloxacin"
                        ], errors="ignore")
    y = df["Tac nhan"]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_cols = load_and_train()

st.markdown("### 📋 Nhập dữ liệu lâm sàng")
user_input = {}

user_input["Tuoi"] = st.number_input("Tuoi", key="input_0")
user_input["Benh ngay thu truoc khi nhap vien"] = st.radio("Benh ngay thu truoc khi nhap vien", ["Không", "Có"], horizontal=True, key="input_1") == "Có"
user_input["Sot"] = st.radio("Sot", ["Không", "Có"], horizontal=True, key="input_2") == "Có"
user_input["Ho"] = st.radio("Ho", ["Không", "Có"], horizontal=True, key="input_3") == "Có"
user_input["Non"] = st.radio("Non", ["Không", "Có"], horizontal=True, key="input_4") == "Có"
user_input["Tieu chay"] = st.radio("Tieu chay", ["Không", "Có"], horizontal=True, key="input_5") == "Có"
user_input["Kich thich"] = st.radio("Kich thich", ["Không", "Có"], horizontal=True, key="input_6") == "Có"
user_input["Tho ren, nhanh"] = st.radio("Tho ren, nhanh", ["Không", "Có"], horizontal=True, key="input_7") == "Có"
user_input["Bo an"] = st.radio("Bo an", ["Không", "Có"], horizontal=True, key="input_8") == "Có"
user_input["Chay mui"] = st.radio("Chay mui", ["Không", "Có"], horizontal=True, key="input_9") == "Có"
user_input["Dam"] = st.radio("Dam", ["Không", "Có"], horizontal=True, key="input_10") == "Có"
user_input["Kho tho"] = st.radio("Kho tho", ["Không", "Có"], horizontal=True, key="input_11") == "Có"
user_input["Kho khe"] = st.radio("Kho khe", ["Không", "Có"], horizontal=True, key="input_12") == "Có"
user_input["Ran phoi"] = st.radio("Ran phoi", ["Không", "Có"], horizontal=True, key="input_13") == "Có"
user_input["Dong dac"] = st.radio("Dong dac", ["Không", "Có"], horizontal=True, key="input_14") == "Có"
user_input["Co lom long nguc"] = st.radio("Co lom long nguc", ["Không", "Có"], horizontal=True, key="input_15") == "Có"
user_input["Nhip tho"] = st.number_input("Nhip tho", key="input_16")
user_input["Mach"] = st.number_input("Mach", key="input_17")
user_input["SpO2"] = st.radio("SpO2", ["Không", "Có"], horizontal=True, key="input_18") == "Có"
user_input["Nhiet do"] = st.number_input("Nhiet do", key="input_19")
user_input["CRP"] = st.number_input("CRP", key="input_20")
user_input["Bach cau"] = st.number_input("Bach cau", key="input_21")
user_input["Sử dụng kháng sinh trước khi đến viện"] = st.radio("Sử dụng kháng sinh trước khi đến viện", ["Không", "Có"], horizontal=True, key="input_22") == "Có"


if st.button("🔍 Dự đoán"):
    df_input = pd.DataFrame([user_input])
    for c in df_input.columns:
        if isinstance(df_input[c][0], bool):
            df_input[c] = df_input[c].astype(int)
    prediction = model.predict(df_input)[0]
    st.success(f"Tác nhân dự đoán: **{prediction}**")

    khang_sinh = {'H. influenzae': ['Amoxicilin clavulanic', 'Ceftriaxone'], 'K. pneumonia': ['Meropenem', 'Ceftriaxone'], 'M. catarrhalis': ['Amoxicilin clavulanic', 'Clarithromycin'], 'M. pneumonia': ['Clarithromycin', 'Levofloxacin'], 'RSV': [], 'S. aureus': ['Vancomycin', 'Clindamycin'], 'S. epidermidis': ['Vancomycin'], 'S. mitis': ['Penicillin'], 'S. pneumonia': ['Ceftriaxone', 'Vancomycin'], 'unspecified': []}
    abx = khang_sinh.get(prediction, [])
    st.markdown("### 💊 Kháng sinh gợi ý:")
    if abx:
        for i in abx:
            st.write(f"- **{i}**")
    else:
        st.info("Không có kháng sinh nào được gợi ý.")
