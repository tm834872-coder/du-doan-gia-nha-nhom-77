"""
house_model.py - Loi xu ly cho app demo (Nhom 7 - Du doan gia nha)

- Hoi quy tuyen tinh (Linear Regression): du doan GiaBan (USD)
- K-means: xep ngoi nha vao 1 phan khuc (cum) dua tren dac diem chinh

Huan luyen tu file nhom7_GiaNha_train.xlsx, danh gia tren nhom7_GiaNha_test.xlsx.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
TRAIN_FILE = HERE / "nhom7_GiaNha_train.xlsx"
TEST_FILE = HERE / "nhom7_GiaNha_test.xlsx"
TARGET = "GiaBan"

# ---- Thuoc tinh dung cho Linear Regression (ban rut gon, de nhap tay) ----
LR_NUMERIC = [
    "ChatLuongTongThe",      # chat luong tong the (1-10)
    "DienTichTang1",         # dien tich tang 1 (ft2)
    "DienTichTang2",         # dien tich tang 2 (ft2)
    "NamXayDung",            # nam xay dung
    "TongDienTichMong",      # tong dien tich tang ham (ft2)
    "SucuongNhaXe_Oto",      # suc chua nha xe (so oto)
    "DienTichLoDat",         # dien tich lo dat (ft2)
    "PhongTamDayDu",         # so phong tam day du
    "PhongNguTrenMatDat",    # so phong ngu
    "TongSoPhongTrenMatDat", # tong so phong tren mat dat
]
KITCHEN_DUMMIES = ["ChatLuongBap_Fa", "ChatLuongBap_Gd", "ChatLuongBap_TA"]  # mac dinh = Ex
KITCHEN_LABELS = {"Ex": None, "Gd": "ChatLuongBap_Gd", "TA": "ChatLuongBap_TA", "Fa": "ChatLuongBap_Fa"}

# ---- Thuoc tinh dung cho K-means ----
KM_FEATURES = [
    "ChatLuongTongThe", "DienTichTrenMatDat", "NamXayDung", "TongDienTichMong",
    "SucuongNhaXe_Oto", "DienTichLoDat", "PhongTamDayDu", "TongSoPhongTrenMatDat",
]
N_CLUSTERS = 3
CLUSTER_NAMES = ["Phân khúc phổ thông", "Phân khúc trung cấp", "Phân khúc cao cấp"]


@dataclass
class HouseModels:
    lr: LinearRegression
    lr_columns: list
    km: KMeans
    km_scaler: StandardScaler
    cluster_rank: dict            # nhan cum goc -> thu tu (0 = re nhat)
    cluster_stats: pd.DataFrame   # gia trung binh / so mau moi cum
    neighborhoods: list
    metrics: dict
    price_range: tuple
    feature_ranges: dict = field(default_factory=dict)


def _load(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def train_models() -> HouseModels:
    tr, te = _load(TRAIN_FILE), _load(TEST_FILE)
    neigh_cols = [c for c in tr.columns if c.startswith("KhuVuc_")]
    lr_cols = LR_NUMERIC + KITCHEN_DUMMIES + neigh_cols

    Xtr, Xte = tr[lr_cols].astype(float), te[lr_cols].astype(float)
    lr = LinearRegression().fit(Xtr, tr[TARGET])
    pred = lr.predict(Xte)
    metrics = {
        "MAE": mean_absolute_error(te[TARGET], pred),
        "RMSE": mean_squared_error(te[TARGET], pred) ** 0.5,
        "R2": r2_score(te[TARGET], pred),
        "n_train": len(tr),
        "n_test": len(te),
    }

    scaler = StandardScaler().fit(tr[KM_FEATURES].astype(float))
    Z = scaler.transform(tr[KM_FEATURES].astype(float))
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit(Z)
    price_by_cluster = tr.groupby(km.labels_)[TARGET].mean().sort_values()
    rank = {int(c): i for i, c in enumerate(price_by_cluster.index)}
    stats = (
        tr[[TARGET]].copy().assign(_c=[rank[int(l)] for l in km.labels_])
        .groupby("_c")
        .agg(so_nha=(TARGET, "size"), gia_tb=(TARGET, "mean"), gia_trung_vi=(TARGET, "median"))
        .rename(index=lambda i: CLUSTER_NAMES[i])
    )

    ranges = {c: (float(tr[c].min()), float(tr[c].max())) for c in LR_NUMERIC}
    return HouseModels(
        lr=lr, lr_columns=lr_cols, km=km, km_scaler=scaler, cluster_rank=rank,
        cluster_stats=stats,
        neighborhoods=[c.replace("KhuVuc_", "") for c in neigh_cols],
        metrics=metrics,
        price_range=(float(tr[TARGET].min()), float(tr[TARGET].max())),
        feature_ranges=ranges,
    )


def predict(models: HouseModels, inp: dict) -> dict:
    """inp: chat_luong, tang1, tang2, nam_xay, tang_ham, oto, lo_dat, tam, ngu, tong_phong, bep, khu_vuc"""
    row = pd.Series(0.0, index=models.lr_columns)
    row["ChatLuongTongThe"] = inp["chat_luong"]
    row["DienTichTang1"] = inp["tang1"]
    row["DienTichTang2"] = inp["tang2"]
    row["NamXayDung"] = inp["nam_xay"]
    row["TongDienTichMong"] = inp["tang_ham"]
    row["SucuongNhaXe_Oto"] = inp["oto"]
    row["DienTichLoDat"] = inp["lo_dat"]
    row["PhongTamDayDu"] = inp["tam"]
    row["PhongNguTrenMatDat"] = inp["ngu"]
    row["TongSoPhongTrenMatDat"] = inp["tong_phong"]
    col = KITCHEN_LABELS.get(inp["bep"])
    if col:
        row[col] = 1.0
    nb = "KhuVuc_" + inp["khu_vuc"]
    if nb in row.index:
        row[nb] = 1.0
    price = float(models.lr.predict(row.to_frame().T)[0])

    km_row = pd.DataFrame([{
        "ChatLuongTongThe": inp["chat_luong"],
        "DienTichTrenMatDat": inp["tang1"] + inp["tang2"],
        "NamXayDung": inp["nam_xay"],
        "TongDienTichMong": inp["tang_ham"],
        "SucuongNhaXe_Oto": inp["oto"],
        "DienTichLoDat": inp["lo_dat"],
        "PhongTamDayDu": inp["tam"],
        "TongSoPhongTrenMatDat": inp["tong_phong"],
    }])[KM_FEATURES]
    raw = int(models.km.predict(models.km_scaler.transform(km_row))[0])
    rank = models.cluster_rank[raw]
    return {"price": price, "cluster_rank": rank, "cluster_name": CLUSTER_NAMES[rank]}
