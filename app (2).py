# -*- coding: utf-8 -*-
"""
APP DEMO - DU DOAN GIA NHA (NHOM 7)
Chay:  streamlit run app.py
Can cung thu muc: house_model.py, nhom7_GiaNha_train.xlsx, nhom7_GiaNha_test.xlsx
"""
import streamlit as st

import house_model as hm

st.set_page_config(page_title="Dự đoán giá nhà - Nhóm 7", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="Đang huấn luyện mô hình từ file train...")
def get_models():
    return hm.train_models()


if not hm.TRAIN_FILE.exists() or not hm.TEST_FILE.exists():
    st.error("Thiếu file dữ liệu. Hãy đặt nhom7_GiaNha_train.xlsx và nhom7_GiaNha_test.xlsx cùng thư mục với app.py.")
    st.stop()

models = get_models()
m = models.metrics

st.title("🏠 Dự đoán giá nhà")
st.caption(
    "Nhóm 7 · Hồi quy tuyến tính dự đoán giá bán · K-means xếp nhà vào phân khúc. "
    "Dữ liệu: bộ nhà ở Ames (Iowa, Mỹ) – giá tính bằng USD, diện tích tính bằng ft² (1 m² ≈ 10,76 ft²)."
)

with st.sidebar:
    st.header("Độ chính xác trên tập test")
    st.metric("R²", f"{m['R2']:.3f}")
    st.metric("MAE (sai số tuyệt đối TB)", f"{m['MAE']:,.0f} USD")
    st.metric("RMSE", f"{m['RMSE']:,.0f} USD")
    st.caption(f"Huấn luyện trên {m['n_train']} mẫu, kiểm tra trên {m['n_test']} mẫu (80:20).")
    st.divider()
    st.caption(
        "Lưu ý: app dùng bản rút gọn (các thuộc tính chính) của mô hình để người dùng dễ nhập; "
        "kết quả chỉ mang tính tham khảo."
    )

st.subheader("1. Nhập thông tin ngôi nhà")
c1, c2, c3 = st.columns(3)

with c1:
    chat_luong = st.slider("Chất lượng tổng thể (1 = kém, 10 = rất tốt)", 1, 10, 6)
    nam_xay = st.number_input("Năm xây dựng", min_value=1870, max_value=2025, value=1973, step=1)
    bep = st.selectbox(
        "Chất lượng nhà bếp",
        ["Ex", "Gd", "TA", "Fa"],
        index=2,
        format_func=lambda x: {"Ex": "Xuất sắc", "Gd": "Tốt", "TA": "Trung bình", "Fa": "Khá"}[x],
    )
    khu_vuc = st.selectbox(
        "Khu vực (Neighborhood)",
        ["Blmngtn"] + models.neighborhoods,
        index=1 + models.neighborhoods.index("NAmes"),
    )

with c2:
    tang1 = st.number_input("Diện tích tầng 1 (ft²)", min_value=300, max_value=5000, value=1087, step=10)
    tang2 = st.number_input("Diện tích tầng 2 (ft²) – 0 nếu không có", min_value=0, max_value=2500, value=0, step=10)
    tang_ham = st.number_input("Tổng diện tích tầng hầm (ft²) – 0 nếu không có", min_value=0, max_value=6500, value=990, step=10)
    lo_dat = st.number_input("Diện tích lô đất (ft²)", min_value=1000, max_value=220000, value=9600, step=100)

with c3:
    ngu = st.number_input("Số phòng ngủ", min_value=0, max_value=8, value=3, step=1)
    tam = st.number_input("Số phòng tắm đầy đủ", min_value=0, max_value=4, value=2, step=1)
    tong_phong = st.number_input("Tổng số phòng (không tính phòng tắm)", min_value=2, max_value=14, value=6, step=1)
    oto = st.number_input("Sức chứa nhà xe (số ô tô)", min_value=0, max_value=4, value=2, step=1)

if st.button("Dự đoán giá", type="primary"):
    res = hm.predict(models, dict(
        chat_luong=chat_luong, tang1=tang1, tang2=tang2, nam_xay=nam_xay, tang_ham=tang_ham,
        oto=oto, lo_dat=lo_dat, tam=tam, ngu=ngu, tong_phong=tong_phong, bep=bep, khu_vuc=khu_vuc,
    ))
    lo, hi = models.price_range

    st.subheader("2. Kết quả")
    r1, r2 = st.columns(2)
    if res["price"] < lo:
        r1.metric("Giá dự đoán (Hồi quy tuyến tính)", "Ngoài phạm vi mô hình")
        st.warning(
            f"Cấu hình này cho giá dự đoán thấp hơn mức nhỏ nhất trong dữ liệu huấn luyện ({lo:,.0f} USD). "
            "Hãy kiểm tra lại số liệu nhập."
        )
    else:
        r1.metric("Giá dự đoán (Hồi quy tuyến tính)", f"{res['price']:,.0f} USD")
        r1.caption(f"Sai số trung bình khoảng ±{m['MAE']:,.0f} USD (MAE trên tập test).")
    r2.metric("Phân khúc (K-means)", res["cluster_name"])

    st.markdown("**Giá trung bình của từng phân khúc trong dữ liệu huấn luyện**")
    stats = models.cluster_stats.copy()
    stats.columns = ["Số nhà", "Giá trung bình (USD)", "Giá trung vị (USD)"]
    st.dataframe(stats.style.format({"Giá trung bình (USD)": "{:,.0f}", "Giá trung vị (USD)": "{:,.0f}"}), use_container_width=True)

    if not (lo <= res["price"] <= hi * 1.0):
        st.info("Giá dự đoán nằm ngoài khoảng giá trong dữ liệu huấn luyện, nên độ tin cậy thấp hơn.")
else:
    st.info("Điền thông tin rồi bấm **Dự đoán giá**.")
