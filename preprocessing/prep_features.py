import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import EmbeddingBlock, FourierVectorize

# 노트북에서 쓰던 상수 복사
T     = 288
LOW_K = int(T * 0.10)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 벡터화 역함수
def vectorize(gamma: torch.Tensor) -> torch.Tensor:
    B, K = gamma.shape
    k = (K - 1) // 2
    R0 = gamma[:, :1]
    R  = gamma[:, 1 : k+1]
    I  = gamma[:, k+1 : ]
    half = torch.complex(R, I)
    conj_half = torch.conj(torch.flip(half[:, 1:], dims=[1]))
    full_spec = torch.cat([R0, half, conj_half], dim=1)
    return torch.fft.ifft(full_spec, n=T).real

# 노트북 원본 전처리 스크립트
def build_psi():
    X      = np.load("X_T1.npy")
    X_mean = np.load("X_mean.npy")
    X_std  = np.load("X_std.npy")
    Xn     = (X - X_mean) / (X_std + 1e-6)

    embed = EmbeddingBlock(D=16).to(device).eval()
    fvec  = FourierVectorize(T, k_bins=LOW_K, mode="ri").to(device).eval()

    loader = DataLoader(
        TensorDataset(torch.tensor(Xn, dtype=torch.float32)),
        batch_size=512, shuffle=False
    )
    all_psi = []
    with torch.no_grad():
        for (xb,) in loader:
            xb  = xb.to(device)
            H   = embed(xb)
            psi = fvec(H).cpu()
            all_psi.append(psi)
    psi = torch.cat(all_psi, dim=0)
    torch.save(psi, "psi_T1.pt")
    print("✅ Saved pre-computed features:", psi.shape)

if __name__ == "__main__":
    build_psi()
