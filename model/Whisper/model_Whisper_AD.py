import torch
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class Whisper_AD:
    def __init__(self, args):
        super(Whisper_AD, self).__init__()
        self.Wseg = args.Wseg
        self.C = args.C
        
        self.Wwin = args.Wwin
        self.Kc = args.Kc
        self.kmeans = KMeans(n_clusters=self.Kc)

        self.train_r_set = []
        self.train_r_set_sample = []
        self.train_distances = None

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

    def add_train_data(self, src_S):
        for S in src_S:
            r_set = []
            
            v_length = [s_i[0] for s_i in S]
            v_iat_ms = [s_i[1] for s_i in S]
            R_length = self.frequency_domain_analysis(v_length) # Nf, Kf = R.shape
            R_iat = self.frequency_domain_analysis(v_iat_ms) # Nf, Kf = R.shape
            Nf, Kf = R_length.shape
            assert Nf == R_iat.shape[0] and Kf == R_iat.shape[1]
            Wwin = self.Wwin
            if Nf >= Wwin:
                Nt = Nf // Wwin
                for i in range(Nt):
                    l = i * Wwin
                    ri_length = torch.mean(R_length[l: l + Wwin], dim=0).numpy().tolist()
                    ri_iat = torch.mean(R_iat[l: l + Wwin], dim=0).numpy().tolist()
                    ri = ri_length + ri_iat
                    self.train_r_set.append(ri)
                    r_set.append(ri)
            else:
                ri_length = torch.mean(R_length, dim=0).numpy().tolist()
                ri_iat = torch.mean(R_iat, dim=0).numpy().tolist()
                ri = ri_length + ri_iat
                self.train_r_set.append(ri)
                r_set.append(ri)
            self.train_r_set_sample.append(r_set)
            
    def clear_train_data(self):
        self.train_r_set = []
        self.train_r_set_sample = []

    def train(self):
        src = np.array(self.train_r_set)
        self.kmeans.fit(src)
        
        distances = []
        for r_set in self.train_r_set_sample:
            r_set = np.array(r_set)
            _, distances_ri = pairwise_distances_argmin_min(r_set, self.kmeans.cluster_centers_, axis=1)
            distance_sample = np.mean(distances_ri)
            distances.append(distance_sample)
        self.train_distances = distances
        return distances

    def test(self, src_S):
        test_r_set_sample = []
        for S in src_S:
            r_set = []
            
            v_length = [s_i[0] for s_i in S]
            v_iat_ms = [s_i[1] for s_i in S]
            R_length = self.frequency_domain_analysis(v_length) # Nf, Kf = R.shape
            R_iat = self.frequency_domain_analysis(v_iat_ms) # Nf, Kf = R.shape
            Nf, Kf = R_length.shape
            assert Nf == R_iat.shape[0] and Kf == R_iat.shape[1]
            Wwin = self.Wwin
            if Nf >= Wwin:
                Nt = Nf // Wwin
                for i in range(Nt):
                    l = i * Wwin
                    ri_length = torch.mean(R_length[l: l + Wwin], dim=0).numpy().tolist()
                    ri_iat = torch.mean(R_iat[l: l + Wwin], dim=0).numpy().tolist()
                    ri = ri_length + ri_iat
                    r_set.append(ri)
            else:
                ri_length = torch.mean(R_length, dim=0).numpy().tolist()
                ri_iat = torch.mean(R_iat, dim=0).numpy().tolist()
                ri = ri_length + ri_iat
                r_set.append(ri)
            test_r_set_sample.append(r_set)
        
        distances = []
        for r_set in test_r_set_sample:
            r_set = np.array(r_set)
            _, distances_ri = pairwise_distances_argmin_min(r_set, self.kmeans.cluster_centers_, axis=1)
            distance_sample = np.mean(distances_ri)
            distances.append(distance_sample)
        return distances
        