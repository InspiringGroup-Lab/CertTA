import torch
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint


# table for combination number
n = 150
m = 30
C = np.array([0] * ((n + 1) * (m + 1)), dtype='O').reshape(n + 1, m + 1)
C[0][0] = 1
for i in range(1, n + 1):
    for j in range(min(m + 1, i + 1)):
        if j == 0 or j == i:
            C[i][j] = 1
        else:
            C[i][j] = C[i-1][j] + C[i-1][j-1]


def p_A_lower_confidence_bound(p_A, sample_n, alpha=0.001):
    '''
        Return the (1 - alpha) lower confidence bound of p_A
        
        p_A: the frequency of c_A after Monte-Carlo samlpling
        sample_n: the sample number
        alpha: the probability with which the estimation of p_A is wrong
    '''
    return proportion_confint(int(sample_n * p_A), sample_n, 2 * alpha, method='beta')[0]


def radius_packet_deletion(p_A, n, d):
    if d is None or d >= n:
        return 0
    
    n_del = 0
    while C[n - n_del][d] >= 2 * (1 - p_A) * C[n][d]:
        n_del += 1
    return n_del - 1


def radius_packet_insertion(p_A, n, d):
    if d is None or d >= n:
        return 0
    
    n_ins = 0
    while C[n + n_ins][d] <= 2 * p_A * C[n][d]:
        n_ins += 1
    return n_ins - 1


def radius_packet_substitution(p_A, n, d):
    if d is None or d >= n:
        return 0
    
    n_sub = 0
    while C[n - n_sub][d] >= (3 / 2 - p_A) * C[n][d]:
        n_sub += 1
    return n_sub - 1


def radius_joint_exp(p_A, d_sel, beta_length, beta_time_ms, n, yatc=False):
    # return: S_jnt = {(n_del, n_ins, n_sub, r_additive_star)}
    S_jnt = []

    d_sel = d_sel if d_sel is not None else n
    beta_length = int(beta_length * np.sqrt(2 / np.pi)) if beta_length is not None else 0
    beta_time_ms = beta_time_ms * np.sqrt(2 / np.pi) if beta_time_ms is not None else 0

    d = d_sel
    n_del_max = min(radius_packet_deletion(p_A, n, d), n)
    n_ins_max = radius_packet_insertion(p_A, n, d)
    n_sub_max = min(radius_packet_substitution(p_A, n, d), n)

    
    n_del_max = 0 # when there is no packet deletion applied to adversarial flows, n_del_max can be set to 0 to simplify the robustness region
    for n_del in range(n_del_max+1):

        for n_ins in range(n_ins_max+1):

            for n_sub in range(n_sub_max+1):
                
                P1 = 1 - C[n - n_del + n_ins][d] / 2 / C[n][d]
                P2 = 2 - p_A - C[n - n_del - n_sub][d] / C[n][d]
                if P1 < P2:
                    continue
                if beta_length == 0 and beta_time_ms == 0:
                    r_additive_star = 0
                else:
                    r_additive_star = (beta_length + beta_time_ms) * (np.log(P1) - np.log(P2))
                    
                S_jnt.append((n_del, n_ins, n_sub, r_additive_star))
                
    return S_jnt


def radius_gaussian(p_A, sigma_gaussian):
    if sigma_gaussian is None or sigma_gaussian < 0:
        return 0
    R = norm.ppf(p_A) * sigma_gaussian
    return R


def radius_bars(p_A, noise_generator, t):
    radius_norm = norm.ppf(p_A)
    radius_norm_dim = torch.tensor(radius_norm).unsqueeze(0).repeat((1, noise_generator.d)).to(noise_generator.device)
    radius_feat_dim = noise_generator.norm_to_feat(radius_norm_dim).detach().cpu().numpy() * t
    radius_feat = radius_feat_dim.mean(1)
    return radius_feat_dim, float(radius_feat)
    