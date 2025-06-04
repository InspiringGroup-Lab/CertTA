import torch
import numpy as np
from tqdm import tqdm


DIRECTION_LENGTH_IDX = 0
IAT_IDX = 1


def PerturbationFunction(x, 
        add_size_noise, add_delay_noise_ms, attack_beta_length, attack_beta_time_ms, attack_pr_sel, attack_r_additive_star,
        insert_where_noise, insert_size_noise, insert_delay_noise_ms, attack_insert_pkts):
    
    batch_size, _, max_flow_length = x.shape
    direction_length = x[:,DIRECTION_LENGTH_IDX,:]
    
    ##########################################################################
    # Additive Perturbations
    attack_beta_length = int(attack_beta_length * np.sqrt(2 / np.pi))
    attack_beta_time_ms = attack_beta_time_ms * np.sqrt(2 / np.pi)
    # 1. mask the noise by existing packets
    mask = (direction_length != 0)
    true_flow_length = torch.sum(mask, dim=1)
    add_delay_noise_ms = add_delay_noise_ms * mask
    add_size_noise = add_size_noise * mask

    # 2. regulate the delay noise & size noise
    add_noise_weighted_sum = add_size_noise / attack_beta_length * (attack_beta_time_ms + attack_beta_length) + add_delay_noise_ms / attack_beta_time_ms * (attack_beta_time_ms + attack_beta_length)
    rescale_factor = []
    for sample_idx in range(batch_size):
        n = true_flow_length[sample_idx]
        d = int(torch.ceil(attack_pr_sel * n))
        tops = torch.argsort(add_noise_weighted_sum[sample_idx], descending=True)
        top_d = tops[:d]
        current_r_additive_star = torch.sum(add_noise_weighted_sum[sample_idx][top_d])
        rescale_factor.append(attack_r_additive_star / current_r_additive_star)
    rescale_factor = torch.stack(rescale_factor).unsqueeze(1).repeat(1, max_flow_length)
    add_delay_noise_ms = add_delay_noise_ms * rescale_factor
    add_size_noise = add_size_noise * rescale_factor
    
    # 3. only integer padding size is allowed & recover the padding direction
    add_size_noise = torch.floor(add_size_noise)
    add_size_noise = add_size_noise * torch.sign(direction_length)
    
    # 4. concatenate the delay noise & size noise
    lst = [None, None]
    lst[1 - DIRECTION_LENGTH_IDX] = add_delay_noise_ms
    lst[DIRECTION_LENGTH_IDX] = add_size_noise
    add_noise = torch.stack(lst, dim=1)
    
    # 5. apply the add noise to x
    x = x + add_noise
    
    ##########################################################################
    # Insertion Perturbation
    # only integer insertion size is allowed, and the minimum insertion size is 40
    insert_size_noise = torch.floor(insert_size_noise) + torch.full_like(insert_size_noise, 40)

    output = x.clone()
    batch_size, _, max_flow_length = x.shape
    for sample_idx in tqdm(range(batch_size), desc='Sample', disable=True):
        timestamp_ms = x[sample_idx][IAT_IDX]
        iat_ms = [0]
        for pkt_idx in range(1, max_flow_length):
            if timestamp_ms[pkt_idx] != 0:
                iat_ms.append(timestamp_ms[pkt_idx] - timestamp_ms[pkt_idx-1])
            else:
                iat_ms.append(0)
        
        tops = torch.argsort(insert_where_noise[sample_idx], descending=True)
        insert_pos = tops[:attack_insert_pkts]
        
        new_pkt_idx = 0
        original_pkt_idx = 0
        current_ts_ms = x[sample_idx][IAT_IDX][0]
        current_direction = 1
        for noise_pkt_idx in range(max_flow_length):
            if new_pkt_idx >= max_flow_length - 1:
                break

            # insert new packet
            if noise_pkt_idx in insert_pos:
                if new_pkt_idx > 0:
                    current_ts_ms += insert_delay_noise_ms[sample_idx, noise_pkt_idx]
                output[sample_idx, IAT_IDX, new_pkt_idx] = current_ts_ms
                output[sample_idx, DIRECTION_LENGTH_IDX, new_pkt_idx] = current_direction * insert_size_noise[sample_idx, noise_pkt_idx]
                new_pkt_idx += 1
            # insert original packet (packet exists if direction_length != 0)
            elif x[sample_idx][DIRECTION_LENGTH_IDX][original_pkt_idx] != 0:
                current_ts_ms += iat_ms[original_pkt_idx]
                output[sample_idx, IAT_IDX, new_pkt_idx] = current_ts_ms
                output[sample_idx, DIRECTION_LENGTH_IDX, new_pkt_idx] = x[sample_idx][DIRECTION_LENGTH_IDX][original_pkt_idx]
                new_pkt_idx += 1

                original_pkt_idx += 1
                if original_pkt_idx < max_flow_length and x[sample_idx][DIRECTION_LENGTH_IDX][original_pkt_idx] != 0:
                    current_direction = 1 if x[sample_idx][DIRECTION_LENGTH_IDX][original_pkt_idx] > 0 else -1
        
        # fill the flow to max_flow_length
        assert new_pkt_idx <= max_flow_length
        for pkt_idx in range(new_pkt_idx, max_flow_length):
            output[sample_idx, IAT_IDX, pkt_idx] = 0
            output[sample_idx, DIRECTION_LENGTH_IDX, pkt_idx] = 0
    
    return output
    
    
def get_decider():
    return PerturbationFunction
