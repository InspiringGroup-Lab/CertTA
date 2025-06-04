import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import sys
import json
import torch
import argparse
from tqdm import tqdm

sys.path.append('.')
from evaluation.utilities import *
from attack.Blanket.Noiser import *
from attack.Blanket.Decider import *


def load_data(data_file, max_flow_length=100):
    flows = []
    with open(data_file) as fp:
        data = json.load(fp)
    for flow in data:
        assert flow['packet_num'] == len(flow['timestamp'])
        assert len(flow['timestamp']) == len(flow['direction_length'])
        assert flow['timestamp'][0] == 0

        direction_length = flow['direction_length']
        timestamp_ms = [i * 1e3 for i in flow['timestamp']]
        iat_ms = [0] + [timestamp_ms[i] - timestamp_ms[i-1] for i in range(1, flow['packet_num'])]
        packet_num = flow['packet_num']
        LENGTH = max_flow_length
        if packet_num < LENGTH:
            direction_length += [0] * (LENGTH - packet_num)
            iat_ms += [0] * (LENGTH - packet_num)
        else:
            direction_length = direction_length[:LENGTH]
            iat_ms = iat_ms[:LENGTH]
        src_seq = [direction_length, iat_ms]
        tgt = flow['label']
        source_file = flow['source_file']
        flows.append((src_seq, tgt, source_file))
    return flows
 

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    
    # attack parameters
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--attack_beta_length", type=float, default=100)
    parser.add_argument("--attack_beta_time_ms", type=float, default=40)
    parser.add_argument("--attack_pr_sel", type=float, default=0.15)
    parser.add_argument("--attack_r_additive_star", type=float, default=21.958)
    parser.add_argument("--attack_insert_pkts", type=int, default=2)
    
    args = parser.parse_args()
    
    args.attack = 'Blanket'
    args.save_dir = './attack/{}/{}/'.format(args.attack, args.dataset)
    args.save_dir += 'Blanket_beta_length_{}_beta_time_ms_{}_pr_sel_{}_r_additive_star_{}_insert_pkts_{}/'.format(
        args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_sel, args.attack_r_additive_star, args.attack_insert_pkts)
    print('save dir: {}'.format(args.save_dir))
    print('--------------------------------------')
    
    print('Loading the dataset.')
    args.dataset_dir = './dataset/{}/json/'.format(args.dataset)
    with open(args.dataset_dir + 'statistics.json') as fp:
        statistics_json = json.load(fp)
    args.labels_num = statistics_json['label_num']
    args.pcap_level = False
    args.max_flow_length = {'CICDOH20': 100, 'TIISSRC23': 100}[args.dataset]
    
    test_flows = load_data(args.dataset_dir + 'test.json', args.max_flow_length)
    args.test_flows_num = len(test_flows)

    print('Loading the noiser models for Blanket.')
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    add_noiser = AddNoiser(args.max_flow_length, args.hidden_dim)
    load_or_initialize_parameters(add_noiser, args.save_dir + 'add_noiser.bin')
    add_noiser = add_noiser.to(args.device)
    args.attack_beta_length = int(args.attack_beta_length * np.sqrt(2 / np.pi))
    args.attack_beta_time_ms = args.attack_beta_time_ms * np.sqrt(2 / np.pi)

    insert_noiser = InsertNoiser(args.max_flow_length, args.hidden_dim)
    load_or_initialize_parameters(insert_noiser, args.save_dir + 'insert_noiser.bin')
    insert_noiser = insert_noiser.to(args.device)
    
    attack_set = []
    for flow_idx in tqdm(range(args.test_flows_num), desc='Testing flows'):
        src = torch.tensor([test_flows[flow_idx][0]]).to(args.device)
        batch_size, _, max_flow_length = src.shape
        assert batch_size == 1

        ################################
        # Additive Perturbations
        add_size_noise, add_delay_noise_ms = add_noiser(src)

        # x.shape = batch_size, 2, max_flow_length
        x = src
        batch_size, _, max_flow_length = x.shape
        timestamp_ms = x[:,IAT_IDX,:]
        direction_length = x[:,DIRECTION_LENGTH_IDX,:]
        
        # mask the noise by existing packets
        mask = (direction_length != 0)
        true_flow_length = torch.sum(mask, dim=1)
        add_delay_noise_ms = add_delay_noise_ms * mask
        add_size_noise = add_size_noise * mask
        
        # regulate the delay noise & size noise
        add_noise_weighted_sum = add_size_noise / args.attack_beta_length * (args.attack_beta_time_ms + args.attack_beta_length) + add_delay_noise_ms / args.attack_beta_time_ms * (args.attack_beta_time_ms + args.attack_beta_length)
        rescale_factor = []
        for sample_idx in range(batch_size):
            n = true_flow_length[sample_idx]
            d = int(torch.ceil(args.attack_pr_sel * n))
            tops = torch.argsort(add_noise_weighted_sum[sample_idx], descending=True)
            top_d = tops[:d]
            current_r_additive_star = torch.sum(add_noise_weighted_sum[sample_idx][top_d])
            rescale_factor.append(args.attack_r_additive_star / current_r_additive_star)
        rescale_factor = torch.stack(rescale_factor).unsqueeze(1).repeat(1, max_flow_length)
        add_delay_noise_ms = add_delay_noise_ms * rescale_factor
        add_size_noise = add_size_noise * rescale_factor
        
        # only integer padding size is allowed
        add_size_noise = torch.floor(add_size_noise)

        ################################
        # Insertion Perturbation
        insert_where_noise, insert_size_noise, insert_delay_noise_ms = insert_noiser(src)
        # only integer padding size is allowed, and the minimum padding size is 40
        insert_size_noise = torch.floor(insert_size_noise) + torch.full_like(insert_size_noise, 40)
        
        ################################
        # Generating attack.json
        x = src
        sample_idx = 0
        tops = torch.argsort(insert_where_noise[sample_idx], descending=True)
        insert_pos = tops[:args.attack_insert_pkts]

        actions = []
        new_pkt_idx = 0
        original_pkt_idx = 0
        for noise_pkt_idx in range(max_flow_length):
            # insert new packet
            if noise_pkt_idx in insert_pos:
                added_delay = insert_delay_noise_ms[sample_idx, noise_pkt_idx] / 1e3 if new_pkt_idx > 0 else 0
                actions.append(
                    {
                        'action': "inserting",
                        "value": int(insert_size_noise[sample_idx, noise_pkt_idx]),
                        "added_delay": float(added_delay)
                    }
                )
                new_pkt_idx += 1
            # insert original packet (packet exists if direction_length != 0)
            elif x[sample_idx][DIRECTION_LENGTH_IDX][original_pkt_idx] != 0:
                actions.append(
                    {
                        'action': "padding",
                        "value": int(add_size_noise[sample_idx, original_pkt_idx]),
                        "added_delay": float(add_delay_noise_ms[sample_idx, original_pkt_idx] / 1e3)
                    }
                )
                new_pkt_idx += 1
                original_pkt_idx += 1
        
        for pkt_idx in range(original_pkt_idx, max_flow_length):
            if x[sample_idx][DIRECTION_LENGTH_IDX][pkt_idx] == 0:
                break

            actions.append(
                {
                    'action': "padding",
                    "value": int(add_size_noise[sample_idx, pkt_idx]),
                    "added_delay": float(add_delay_noise_ms[sample_idx, pkt_idx] / 1e3)
                }
            )
            new_pkt_idx += 1

        true_flow_length = torch.sum(x[sample_idx, DIRECTION_LENGTH_IDX, :] != 0)
        assert true_flow_length + len(insert_pos) == new_pkt_idx

        attack_set.append(
            {
                'actions': actions,
                'source_file': test_flows[flow_idx][2]
            }
        )
    
    with open(args.save_dir + 'attack.json', 'w') as fp:
        json.dump(attack_set, fp, indent=1)


if __name__ == '__main__':
    main()