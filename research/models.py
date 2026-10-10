# -*- coding: utf-8 -*-
"""
예측 모델 정의.
  - Persistence : 학습 불필요 baseline. "다음 값 = 현재(마지막) 타깃 값".
                  이 기준선을 못 이기면 딥러닝 모델은 의미가 없다.
  - LSTMModel   : 다변량 LSTM (환경+거동 입력 -> 타깃 거동 예측)
  - TransformerModel : 인코더 기반 시계열 Transformer

모든 신경망은 입력 (B, L, F) -> 출력 (B, n_targets) 형태로 통일.
"""
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------
# Baseline: Persistence (naive)
# --------------------------------------------------------------------
class Persistence:
    """
    학습 파라미터 없음. 순수 naive baseline.
    - raw  모드: 예측 = 입력 마지막 스텝의 타깃값 (스케일 공간). 복원하면 '직전값'.
    - diff 모드: 예측 Δ = 0 (변화 없음 가정). 복원하면 역시 '직전값'.
    어느 모드든 "다음 값 = 현재 값" 이라는 동일한 기준선을 의미한다.
    """
    is_torch = False

    def __init__(self, target_idx, mode="raw"):
        self.target_idx = target_idx
        self.mode = mode

    def predict(self, X):
        last = X[:, -1, self.target_idx]         # (N, n_targets)
        if self.mode == "diff":
            return np.zeros_like(last)
        return last


# --------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # 마지막 타임스텝
        out = self.dropout(out)
        return self.fc(out)


# --------------------------------------------------------------------
# Transformer (encoder-only)
# --------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = h[:, -1, :]               # 마지막 타임스텝 표현
        h = self.dropout(h)
        return self.fc(h)


def build_model(name, n_features, n_targets, target_idx, mcfg, target_mode="raw"):
    """설정 이름으로 모델 인스턴스 생성."""
    name = name.lower()
    if name == "persistence":
        return Persistence(target_idx, mode=target_mode)
    if name == "lstm":
        return LSTMModel(n_features, n_targets,
                         hidden_size=mcfg.get("hidden_size", 64),
                         num_layers=mcfg.get("num_layers", 2),
                         dropout=mcfg.get("dropout", 0.2))
    if name == "transformer":
        return TransformerModel(n_features, n_targets,
                                d_model=mcfg.get("d_model", 64),
                                nhead=mcfg.get("nhead", 4),
                                num_layers=mcfg.get("num_layers", 2),
                                dim_feedforward=mcfg.get("dim_feedforward", 128),
                                dropout=mcfg.get("dropout", 0.2))
    raise ValueError(f"알 수 없는 모델: {name}")
