import torch
import torch.nn as nn
import torch.fft
from Shanghai_Datasets import PAST_LEN, FUTURE_LEN

class BSpline1D(nn.Module):
    def __init__(self, num_ctrl_pts, degree=3):
        super().__init__()
        m, p = num_ctrl_pts, degree
        knots = torch.cat([torch.zeros(p+1), torch.linspace(0, 1, max(m-p-1, 1)), torch.ones(p+1)])
        self.register_buffer('knots', knots)
        self.coeff = nn.Parameter(torch.zeros(m))
        self.m, self.p = m, p
    def forward(self, x):
        x = x.view(-1)
        B = x.size(0)
        m, p, knots = self.m, self.p, self.knots
        t0, t1 = knots[:-1][:m], knots[1:][:m]
        Bf = ((x.unsqueeze(1) >= t0) & (x.unsqueeze(1) < t1)).float()
        Bf[:, -1] = ((x >= t0[-1]) & (x <= t1[-1])).float()
        for d in range(1, p+1):
            t_i, t_id = knots[:m], knots[d:d+m]
            denom1 = (t_id - t_i).unsqueeze(0)
            term1 = ((x.unsqueeze(1) - t_i) / (denom1 + 1e-6)) * Bf
            t_ip1, t_idp1 = knots[1:m+1], knots[d+1:d+m+1]
            denom2 = (t_idp1 - t_ip1).unsqueeze(0)
            Bf_shift = torch.cat([Bf[:,1:], torch.zeros(B,1,device=x.device)], dim=1)
            term2 = ((t_idp1 - x.unsqueeze(1)) / (denom2 + 1e-6)) * Bf_shift
            Bf = term1 + term2
        return Bf @ self.coeff

class KANLayer(nn.Module):
    def __init__(self, in_feats, out_feats, G=5, degree=3):
        super().__init__()
        self.splines = nn.ModuleList([BSpline1D(G, degree) for _ in range(in_feats)])
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats))
        self.bias   = nn.Parameter(torch.zeros(out_feats))
    def forward(self, x):
        phi = torch.stack([s(x[:,i]) for i,s in enumerate(self.splines)], dim=1)
        return phi @ self.weight + self.bias

class FKAN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, G=5, degree=3):
        super().__init__()
        self.l1 = KANLayer(in_feats, hid_feats, G, degree)
        self.l2 = KANLayer(hid_feats, out_feats, G, degree)
    def forward(self, psi):
        return self.l2(self.l1(psi))

class EmbeddingBlock(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.beta = nn.Parameter(torch.randn(1, D))
    def forward(self, X):
        return X.unsqueeze(-1) * self.beta

class FourierVectorize(nn.Module):
    def __init__(self, T, k_bins=None, mode="ri"):
        super().__init__()
        self.n_fft  = T
        self.k_bins = k_bins or (T//2 + 1)
        self.mode   = mode
    def forward(self, H):
        B, N, T, D = H.shape
        C = N * D
        F = torch.fft.rfft(H.permute(0,1,3,2).reshape(B, C, T), n=self.n_fft, dim=-1)[:, :, :self.k_bins]
        if self.mode == "ri":
            feat = torch.cat([F.real, F.imag], dim=-1)
        elif self.mode == "mag":
            feat = torch.abs(F)
        else:
            feat = torch.log1p(torch.abs(F))
        return feat.reshape(B, -1)
    def output_dim(self, N, D):
        base = N * D * self.k_bins
        return base * 2 if self.mode == "ri" else base

class InverseFourier(nn.Module):
    def __init__(self, N, D, tau, k_bins):
        super().__init__()
        self.N, self.D, self.tau = N, D, tau
        self.k_bins = k_bins
    def forward(self, gamma):
        B = gamma.size(0)
        C = self.N * self.D
        stacked = gamma.view(B, C, self.k_bins, 2)
        real = stacked[..., 0]
        imag = stacked[..., 1]
        spec = torch.complex(real, imag)
        t = torch.fft.irfft(spec, n=self.tau, dim=-1)
        t = t.view(B, self.N, self.D, self.tau)
        bg = t[:, 0]
        y_hat = bg.mean(dim=1)
        return y_hat

class SugarNet(nn.Module):
    def __init__(self, N=4, D=16, H=64, G=5):
        super().__init__()
        T = PAST_LEN
        tau = FUTURE_LEN
        LOWPASS_K = int(0.1 * T)
        self.embed = EmbeddingBlock(D)
        self.fvec  = FourierVectorize(T, k_bins=LOWPASS_K, mode="ri")
        in_dim  = self.fvec.output_dim(N, D)
        out_dim = N * D * LOWPASS_K * 2
        self.fkan = FKAN(in_dim, H, out_dim, G)
        self.inv  = InverseFourier(N, D, tau, LOWPASS_K)
    def forward(self, X):
        H = self.embed(X)
        psi = self.fvec(H)
        gamma = self.fkan(psi)
        y_hat = self.inv(gamma)
        return y_hat
