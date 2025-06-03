import torch
import numpy as np
from math import log2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class Whisper:
    def __init__(self, args):
        super(Whisper, self).__init__()
        self.Wseg = args.Wseg
        self.C = args.C

        self.truncate_length = 60
        self.parameters = {
            'n_estimators': [20, 50, 100],
            'max_depth': [10, 30, None],
            'max_features': ['sqrt', 'log2'],
            'oob_score': [True, False]
        }
        self.rf = GridSearchCV(RandomForestClassifier(), self.parameters, n_jobs=-1, cv=3, verbose=2)
        self.train_src = []
        self.train_tgt = []

    def weight_transform(self, S):
        # S = [s_1, s_2, ..., s_M]
        # s_i = [length_i, iat_i]
        v = []
        for s_i in S:
            pkt_length = s_i[0]
            pkt_iat_s = s_i[1] / 1e3
            pkt_iat_s = max(pkt_iat_s, 1e-5)
            v_i = pkt_length * 10 - log2(pkt_iat_s) * 15.68
            v.append(v_i)
        return v
    
    def frequency_domain_analysis(self, v):
        Wseg = self.Wseg
        if len(v) < Wseg:
            v = v + [0] * (Wseg - len(v))
        ten = torch.FloatTensor(v)
        
        # DFT on flow vector
        ten_fft = torch.stft(ten, n_fft=Wseg, hop_length=Wseg, center=False, return_complex=False) 
        # calculate the power
        ten_fft_permute = ten_fft.permute(2, 0, 1)
        ten_power = ten_fft_permute[0] * ten_fft_permute[0] + ten_fft_permute[1] * ten_fft_permute[1]
        # log linear transformation
        ten_res = ((ten_power + 1).log2() / self.C).permute(1, 0) 

        # erase the inf and nan
        ten_res = torch.where(torch.isnan(ten_res), torch.full_like(ten_res, 0), ten_res)
        ten_res = torch.where(torch.isinf(ten_res), torch.full_like(ten_res, 0), ten_res)
        
        Nf = len(v) // Wseg
        Kf = Wseg // 2 + 1
        assert ten_res.shape == torch.Size([Nf, Kf])
        return ten_res

    def add_train_data(self, src_S, tgt):
        for S in src_S:
            ###### rf
            v_length = [s_i[0] for s_i in S]
            v_iat_ms = [s_i[1] for s_i in S]
            R_length = self.frequency_domain_analysis(v_length).numpy().reshape(-1).tolist() # Nf, Kf = R.shape
            R_iat = self.frequency_domain_analysis(v_iat_ms).numpy().reshape(-1).tolist() # Nf, Kf = R.shape
            assert len(R_length) == len(R_iat)
            if len(R_length) < self.truncate_length:
                R_length = R_length + [0] * (self.truncate_length - len(R_length))
                R_iat = R_iat + [0] * (self.truncate_length - len(R_iat))
            else:
                R_length = R_length[:self.truncate_length]
                R_iat = R_iat[:self.truncate_length]
            R = R_length + R_iat
            self.train_src.append(R)
            
        self.train_tgt.extend(tgt)

    def clear_train_data(self):
        self.train_tgt = []
        self.train_src = []

    def train(self):
        src, tgt = np.array(self.train_src), np.array(self.train_tgt)
        self.rf.fit(src, tgt)
        print('Best parameters', self.rf.best_params_)
        y_pred = self.rf.predict(src)
        
        return tgt, y_pred

    def test(self, src_S):
        src = []
        for S in src_S:
            v_length = [s_i[0] for s_i in S]
            v_iat_ms = [s_i[1] for s_i in S]
            R_length = self.frequency_domain_analysis(v_length).numpy().reshape(-1).tolist() # Nf, Kf = R.shape
            R_iat = self.frequency_domain_analysis(v_iat_ms).numpy().reshape(-1).tolist() # Nf, Kf = R.shape
            assert len(R_length) == len(R_iat)
            if len(R_length) < self.truncate_length:
                R_length = R_length + [0] * (self.truncate_length - len(R_length))
                R_iat = R_iat + [0] * (self.truncate_length - len(R_iat))
            else:
                R_length = R_length[:self.truncate_length]
                R_iat = R_iat[:self.truncate_length]
            R = R_length + R_iat
            src.append(R)
        src = np.array(src)
        y_pred = self.rf.predict(src)
            
        return y_pred
