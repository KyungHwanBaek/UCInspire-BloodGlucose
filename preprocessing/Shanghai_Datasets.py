import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

PAST_LEN = 288
FUTURE_LEN = 12 
GLOBAL_BG: list[float] = []


def _build_basal_quorum(df: pd.DataFrame, freq: str = "5min") -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Time" in df.columns:
            df.index = pd.to_datetime(df["Time"])
        else:
            raise ValueError("DataFrame must contain a Time column or DatetimeIndex")
    basal_ff = df["Basal"].ffill()
    basal_resamp = basal_ff.resample(freq).ffill()
    bucket_key = basal_resamp.index.time
    bucket_df = pd.DataFrame({"rate": basal_resamp.values, "bucket": bucket_key})
    mode_rate = (
        bucket_df
        .groupby("bucket")["rate"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)
    )
    return mode_rate


def _apply_basal_quorum(df: pd.DataFrame, modal: pd.Series) -> pd.Series:
    rates = []
    for t in df.index:
        rates.append(modal.get(t.time(), 0))
    return pd.Series(rates, index=df.index)


def load_and_process(file_path: str) -> np.ndarray:
    if file_path.endswith(".xls"):
        df = pd.read_excel(file_path, engine="xlrd")
    else:
        df = pd.read_excel(file_path)

    rename = {
        "CGM (mg / dl)": "BG",
        "Basal Rate (U/h)": "Basal",
        "Bolus Insulin (U)": "Bolus",
        "CHO Intake (g)": "CHO",
    }
    df.rename(columns=rename, inplace=True)

    for col in ["BG", "Basal", "Bolus", "CHO"]:
        if col not in df.columns:
            df[col] = 0

    if "Time" in df.columns and df['Basal'].notna().any():
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        modal_basal = _build_basal_quorum(df)
        df['Basal'] = df['Basal'].replace(0, np.nan)
        df['Basal'] = _apply_basal_quorum(df, modal_basal)
    else:
        # no Time or no basal data: forward-fill or zero
        df['Basal'] = df['Basal'].fillna(0)

    # sparse features if possible
    if "Time" in df.columns and df["Basal"].notna().any():
        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
        modal_basal = _build_basal_quorum(df)
        df["Basal"] = df["Basal"].replace(0, np.nan)
        df["Basal"] = _apply_basal_quorum(df, modal_basal)
    else:
        df["Basal"] = df["Basal"].fillna(0)

    for col in ("CHO", "Bolus"):
        df[col] = df[col].fillna(0)
        df[col] = np.log1p(df[col])

    df["Basal"] = np.log1p(df["Basal"])
    GLOBAL_BG.extend(df["BG"].tolist())

    arr = df[["BG", "Basal", "Bolus", "CHO"]].to_numpy().T
    return arr

def generate_sequences(arr: np.ndarray, past_len: int = PAST_LEN, future_len: int = FUTURE_LEN):
    X, Y = [], []
    for i in range(arr.shape[1] - past_len - future_len + 1):
        x = arr[:, i : i + past_len]
        bg_now = arr[0, i + past_len - 1]
        future_bg = arr[0, i + past_len : i + past_len + future_len]
        y = future_bg - bg_now  # ΔBG
        X.append(x)
        Y.append(y)
    return np.stack(X), np.stack(Y)

class SugarDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_dataset_from_folder(folder_path: str):
    files = glob.glob(os.path.join(folder_path, "*.xls")) + glob.glob(os.path.join(folder_path, "*.xlsx"))
    all_X, all_Y = [], []
    for fp in files:
        arr = load_and_process(fp)
        X, Y = generate_sequences(arr)
        if len(X) == 0:
            print(f"⚠️ skip (too short): {fp}")
            continue
        all_X.append(X)
        all_Y.append(Y)

    if not all_X:
        raise ValueError(f"No usable sequences found in {folder_path}")

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    mu, sigma = np.mean(GLOBAL_BG), np.std(GLOBAL_BG)
    X[:, 0, :] = (X[:, 0, :] - mu) / sigma

    return X, Y, mu, sigma

T1_FOLDER = "./Shanghai_Datasets_T1"
X, Y, mu, sigma = get_dataset_from_folder(T1_FOLDER)

np.save("X_T1.npy", X)
np.save("Y_T1.npy", Y)
np.save("X_mean.npy", mu)
np.save("X_std.npy", sigma)

print("X_T1.npy, Y_T1.npy 저장:", X.shape, Y.shape)





