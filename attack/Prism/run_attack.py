import json
from copy import deepcopy
import os
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
import torch

def load_data(args, path):
    data = []
    with open(path, "r") as f:
        raw_data = json.load(f)
        for flow in raw_data:
            packets = []
            length = flow["packet_num"]
            cur_timestamp = 0
            for idx in range(length):
                pkt_direction = 1 if flow["direction_length"][idx] > 0 else -1
                pkt_length = abs(flow["direction_length"][idx])
                src_file = flow["source_file"]
                pkt_iat = flow["timestamp"][idx] - cur_timestamp
                cur_timestamp = flow["timestamp"][idx]
                packets.append([pkt_direction, pkt_length, pkt_iat, 0, idx])
            data.append({
                "packets": packets,
                "source_file": flow["source_file"],
                "label": flow["label"]
            })
    return data
                
                
def temporal_feature_learning(args, train_data):
    feature_count = {}
    for flow in train_data:
        label = flow["label"]
        if label not in feature_count.keys():
            feature_count[label] = {}
        for pkt in flow["packets"]:
            # ignore packets with direction -1
            pkt_direction = pkt[0]
            
            # FIXME: whether to ignore packets with direction -1
            # if pkt_direction == -1:
            #     continue

            pkt_length = pkt[1]
            if pkt_length not in feature_count[label].keys():
                feature_count[label][pkt_length] = 0
            feature_count[label][pkt_length] += 1
    # sort features by frequency
    # basic_feature_format: [bf_value, bf_count]
    basic_features_count = {}
    for label in feature_count.keys():
        basic_features_count[label] = sorted(feature_count[label].items(), key=lambda x: x[1], reverse=True)
    
    # select top 20 features
    basic_feature = {}
    for label in basic_features_count.keys():
        max_feature = sorted(feature_count[label].items(), key=lambda x: x[0], reverse=True)[0][0]
        print("label: ", label)
        print("len(basic_features_count[label]): ", len(basic_features_count[label]))
        print("basic_features_count[label]: ", basic_features_count[label][:args.basic_feature_num])
        print("max_feature: ", max_feature)
        basic_feature[label] = [item[0] for item in basic_features_count[label][:args.basic_feature_num]]
        print("basic_feature: ", basic_feature[label])
        basic_feature[label] = basic_feature[label] + [(max_feature + 4)]
        print("basic_feature: ", basic_feature[label])
        
    return basic_feature

# def temporal_feature_learning(args, train_data):
#     feature_count = {}
#     for flow in train_data:
#         label = flow["label"]
#         if label not in feature_count.keys():
#             feature_count[label] = {}
#         for pkt in flow["packets"]:
#             # ignore packets with direction -1
#             pkt_direction = pkt[0]
#             # if pkt_direction == -1:
#             #     continue

#             pkt_length = pkt[1]
#             if pkt_length not in feature_count[label].keys():
#                 feature_count[label][pkt_length] = 0
#             feature_count[label][pkt_length] += 1
#     # sort features by frequency
#     # basic_feature_format: [bf_value, bf_count]
#     basic_features_count = {}
#     for label in feature_count.keys():
#         basic_features_count[label] = sorted(feature_count[label].items(), key=lambda x: x[1], reverse=True)
    
#     # select top 20 features
#     basic_feature = {}
#     for label in basic_features_count.keys():
#         bf_list = [item[0] for item in basic_features_count[label][:args.basic_feature_num]]
        
#         label_feature = basic_features_count[label]
#         max_bf = sorted(bf_list, reverse=True)[0]
#         remaining_feature = [item for item in label_feature if item[0] > max_bf]
        
#         print("label", label)
#         print("remaining_feature: ", len(remaining_feature))
#         print("remaining_feature: ", remaining_feature)
        
#         remaining_feature = sorted(remaining_feature, key=lambda x: x[1], reverse=True)
        
#         remaining_bf = [item[0] for item in remaining_feature[:args.long_feature_num]]

#         basic_feature[label] = bf_list + remaining_bf
        
#     return basic_feature


# def temporal_feature_learning(args, train_data):
#     feature_count = {}
#     for flow in train_data:
#         label = flow["label"]
#         if label not in feature_count.keys():
#             feature_count[label] = {}
#         for pkt in flow["packets"]:
#             # ignore packets with direction -1
#             pkt_direction = pkt[0]
#             if pkt_direction == -1:
#                 continue

#             pkt_length = pkt[1]
#             if pkt_length not in feature_count[label].keys():
#                 feature_count[label][pkt_length] = 0
#             feature_count[label][pkt_length] += 1
#     # sort features by frequency
#     # basic_feature_format: [bf_value, bf_count]
#     basic_features_count = {}
#     for label in feature_count.keys():
#         basic_features_count[label] = sorted(feature_count[label].items(), key=lambda x: x[0], reverse=False)

#     basic_feature = {}
#     for label in basic_features_count.keys():
#         feature_list = basic_features_count[label]
#         # print("feature_list: ", feature_list)
#         tot_cnt = 0
#         for bf, cnt in feature_list:
#             # print("bf: ", bf)
#             # print("cnt: ", cnt)
#             tot_cnt += cnt
#         bin_size = tot_cnt // args.bin_count
#         feat_bins = []
#         current_bin = {}
#         current_bin_cnt = 0
#         for bf, cnt in feature_list:
#             current_bin[bf] = cnt
#             current_bin_cnt += cnt
#             if current_bin_cnt >= bin_size:
#                 print("current_bin_cnt: ", current_bin_cnt)
#                 feat_bins.append(current_bin)
#                 current_bin = {}
#                 current_bin_cnt = 0
#         if current_bin_cnt > 0:
#             feat_bins.append(current_bin)
#             print("current_bin_cnt: ", current_bin_cnt)
#         bin_bf = []
#         for i in range(len(feat_bins)):
#             feat_bins[i] = sorted(feat_bins[i].items(), key=lambda x: x[1], reverse=True)
#             print("len(feat_bins[i]): ", len(feat_bins[i]))
#             bin_bf += [item[0] for item in feat_bins[i][:args.basic_feature_num // args.bin_count]]
#         basic_feature[label] = bin_bf
#         print(args.basic_feature_num // args.bin_count)
#         print("basic_feature: ", basic_feature[label])
#         assert len(basic_feature[label]) == args.basic_feature_num
#     return basic_feature
    
    
def classify_and_adjust_flows(args, basic_features, data):
    adjusted_data = {}
    for flow in data:
        label = flow["label"]
        if label not in adjusted_data.keys():
            adjusted_data[label] = []
        
        feature_list = [pkt[1] for pkt in flow["packets"]]
        adjusted_feature_list = [Stfeat(feature, basic_features[label]) for feature in feature_list]
        adjusted_data[label].append(adjusted_feature_list)
    
   
    for label in adjusted_data.keys():
        poscnt = 0
        negcnt = 0
        flowcnt = 0
        for adjusted_feature_list in adjusted_data[label]:
            flowcnt += 1
            negcnt += adjusted_feature_list.count(-1)
            poscnt += len(adjusted_feature_list) - adjusted_feature_list.count(-1)
        assert negcnt == 0 , f"Stfeat failed to adjust all features, {label}"
        # print("label: ", label)
        # print("poscnt: ", poscnt)
        # print("negcnt: ", negcnt)
        # print("flowcnt: ", flowcnt)
    return adjusted_data
            
        
def state_transition_modeling(args, adjusted_data, basic_features):
    transition_model = {}
    for label in adjusted_data.keys():
        if label not in transition_model.keys():
            print("creating transition model for label: ", label)
            transition_model[label] = np.zeros((args.max_flow_length - 1, args.basic_feature_num + 1, args.basic_feature_num + 1))
        for t in range(args.max_flow_length - 1):
            
            sample_pool = []
            for flow in adjusted_data[label]:
                if len(flow) < t + 2:
                    continue
                x_t = flow[t]
                x_t_1 = flow[t + 1]
                sample_pool.append((x_t, x_t_1))
            
            for i in range(args.basic_feature_num):
                
                i_sample = [sample for sample in sample_pool if sample[0] == basic_features[label][i]]
                i_sample_cnt = len(i_sample)
                
                for j in range(args.basic_feature_num):
                    j_sample = [sample for sample in i_sample if sample[1] == basic_features[label][j]]
                    j_sample_cnt = len(j_sample)
                    if i_sample_cnt == 0:
                        transition_model[label][t][i][j] = 0
                    else:
                        # print("value: ", j_sample_cnt / i_sample_cnt)
                        transition_model[label][t][i][j] = j_sample_cnt / i_sample_cnt
                        
                        # a = calculate_p(t, i, j, basic_features[label], adjusted_data[label])
                        # if j_sample_cnt / i_sample_cnt != a:
                        #     print("error")
                        # else:
                        #     print("correct")
                    # transition_model[label][t][i][j] = calculate_p(t, i, j, basic_features[label], adjusted_data[label])
                    # a = calculate_p(t, i, j, basic_features[label], adjusted_data[label])
                    # print(a)
                    # transition_model[label][t][i][j] = a
    return transition_model
        
        
    
def untargeted_attack(args, bf_list , transition_model, data):
    res_data = []
    for raw_flow in data:
        flow = deepcopy(raw_flow)
        pkts = deepcopy(flow["packets"])
        label = flow["label"]
        # print("label: ", label)
        # print("bf_list: ", bf_list[label])
        
        # split the flow into packet groups
        pkt_groups = []
        cur_group = [pkts[0]]
        cur_direction = pkts[0][0]
        for i in range(len(pkts) - 1):
            pkt_idx = i + 1
            if pkts[pkt_idx][0] == cur_direction:
                cur_group.append(pkts[pkt_idx])
            else:
                pkt_groups.append({
                    "pkts": cur_group,
                    "direction": cur_direction
                                   })
                cur_group = [pkts[pkt_idx]]
                cur_direction = pkts[pkt_idx][0]
        pkt_groups.append({
            "pkts": deepcopy(cur_group),
            "direction": cur_direction
        })
        
        # print(pkt_groups)
        
        # apply attack to each packet group
        adversarial_pkts = []
        inserted_pkt_cnt = 0
        for pkt_group in pkt_groups:
            
            # ignore packets with direction -1
            # if pkt_group["direction"] == -1:
            #     adversarial_pkts += pkt_group["pkts"]
            #     continue
            
            # get the packets in the group
            pkts = pkt_group["pkts"]
            init_pkt = deepcopy(pkts[0])
            org_pkt_length = init_pkt[1]
            init_pkt[1] = Stfeat(init_pkt[1], bf_list[label], enable_equal=args.enable_equal)
            if args.budget_per_pkt == 0:
                rand_clip = 0
            else:
                rand_clip = np.random.randint(0, int(args.budget_per_pkt/2))
            init_pkt[1] = init_pkt[1] if init_pkt[1] <= org_pkt_length + args.budget_per_pkt else org_pkt_length + args.budget_per_pkt - rand_clip
            # print("diff:" , init_pkt[1] - pkts[0][1], "init_pkt[1]: ", init_pkt[1], "pkts[0][1]: ", pkts[0][1])
            adversarial_pkts.append(init_pkt)
            for idx in range(len(pkts) - 1):
                cur_pkt = deepcopy(pkts[idx+1])
                pos_in_flow = pkts[idx][4]
                if pos_in_flow >= args.max_flow_length - 1:
                    cur_pkt[1] = Stfeat(cur_pkt[1], bf_list[label], enable_equal=args.enable_equal)
                    adversarial_pkts.append(cur_pkt)
                    continue
                transition_matrix = transition_model[label][pos_in_flow]
                i = bf_list[label].index(Stfeat(pkts[idx][1], bf_list[label], enable_exceed=True))
                j_val_list = [transition_matrix[i][j] if(bf_list[label][j] >= cur_pkt[1])else 2 for j in range(args.basic_feature_num + 1)]
                insert_j_val_list = [transition_matrix[i][j] for j in range(args.basic_feature_num + 1)]
                # assert np.argmin(insert_j_val_list) == np.argmin(j_val_list)
                if np.argmin(insert_j_val_list) != np.argmin(j_val_list) and inserted_pkt_cnt < args.max_insert:
                    # assert False
                    inserted_pkt_cnt += 1
                    adversarial_pkts.append([cur_pkt[0], 1600, cur_pkt[2], 1, cur_pkt[4]])
                # print("j_val_list: ", j_val_list)
                # ignore those features that are less than org_pkt_length
                # least_j_val = np.argmin(j_val_list)
                # # print("least_j_val: ", least_j_val)
                # # print("bf_list[label][least_j_val]: ", bf_list[label][least_j_val])
                # cur_pkt[1] = bf_list[label][least_j_val]
                # adversarial_pkts.append(cur_pkt)
                least_prob = np.min(j_val_list)
                if least_prob > 1:
                    adversarial_pkts.append(cur_pkt)
                    continue
                cand_list = [j for j in range(args.basic_feature_num + 1) if j_val_list[j] <= least_prob + args.delta]

                # bf_length_list = [bf_list[label][j] for j in cand_list if bf_list[label][j] <= cur_pkt[1] + args.budget_per_pkt]
                bf_length_list = [bf_list[label][j] for j in cand_list]
                
                # print("least_j_val: ", least_j_val)
                # print("bf_list[label][least_j_val]: ", bf_list[label][least_j_val])
                org_pkt_len = cur_pkt[1]
                
                new_pkt_len = np.min(bf_length_list) if bf_length_list != [] else cur_pkt[1]
                if args.budget_per_pkt == 0:
                    cur_pkt[1] = org_pkt_len
                elif new_pkt_len > org_pkt_len + args.budget_per_pkt:
                    cur_pkt[1] = org_pkt_len + args.budget_per_pkt - np.random.randint(0, int(args.budget_per_pkt/2))
                else:
                    cur_pkt[1] = new_pkt_len
                # cur_pkt[1] = np.min(bf_length_list) if bf_length_list != [] else cur_pkt[1]
                assert cur_pkt[1] >= org_pkt_len, f"cur_pkt[1]: {cur_pkt[1]}, org_pkt_len: {org_pkt_len}"
                
                adversarial_pkts.append(cur_pkt)
        res_data.append({
            "org_packets": flow["packets"],
            "adv_packets": adversarial_pkts,
            "source_file": flow["source_file"],
            "label": label
        })
        # print("res_data: ", res_data)
    return res_data

def targeted_attack(args, bf_list , transition_model, data):
    res_data = []
    for raw_flow in data:
        flow = deepcopy(raw_flow)
        pkts = deepcopy(flow["packets"])
        label = flow["label"]
        # print("label: ", label)
        # print("bf_list: ", bf_list[label])
        
        # split the flow into packet groups
        pkt_groups = []
        cur_group = [pkts[0]]
        cur_direction = pkts[0][0]
        for i in range(len(pkts) - 1):
            pkt_idx = i + 1
            if pkts[pkt_idx][0] == cur_direction:
                cur_group.append(pkts[pkt_idx])
            else:
                pkt_groups.append({
                    "pkts": cur_group,
                    "direction": cur_direction
                                   })
                cur_group = [pkts[pkt_idx]]
                cur_direction = pkts[pkt_idx][0]
        pkt_groups.append({
            "pkts": deepcopy(cur_group),
            "direction": cur_direction
        })
        
        # print(pkt_groups)
        
        # apply attack to each packet group
        adversarial_pkts = []
        for pkt_group in pkt_groups:
            
            # ignore packets with direction -1
            # if pkt_group["direction"] == -1:
            #     adversarial_pkts += pkt_group["pkts"]
            #     continue
            
            # get the packets in the group
            pkts = pkt_group["pkts"]
            init_pkt = deepcopy(pkts[0])
            org_pkt_length = init_pkt[1]
            init_pkt[1] = Stfeat(init_pkt[1], bf_list[label], enable_equal=args.enable_equal)
            init_pkt[1] = init_pkt[1] if init_pkt[1] <= org_pkt_length + args.budget_per_pkt else org_pkt_length + args.budget_per_pkt
            # print("diff:" , init_pkt[1] - pkts[0][1], "init_pkt[1]: ", init_pkt[1], "pkts[0][1]: ", pkts[0][1])
            adversarial_pkts.append(init_pkt)
            for idx in range(len(pkts) - 1):
                cur_pkt = deepcopy(pkts[idx+1])
                pos_in_flow = pkts[idx][4]
                if pos_in_flow >= args.max_flow_length - 1:
                    cur_pkt[1] = Stfeat(cur_pkt[1], bf_list[label], enable_equal=args.enable_equal)
                    adversarial_pkts.append(cur_pkt)
                    continue
                transition_matrix = transition_model[label][pos_in_flow]
                i = bf_list[label].index(Stfeat(pkts[idx][1], bf_list[label], enable_exceed=True))
                j_val_list = [transition_matrix[i][j] if bf_list[label][j] >= cur_pkt[1] else 2 for j in range(args.basic_feature_num + 1)]
                # print("j_val_list: ", j_val_list)
                # ignore those features that are less than org_pkt_length
                # least_j_val = np.argmin(j_val_list)
                least_prob = np.min(j_val_list)
                cand_list = [j for j in range(args.basic_feature_num + 1) if j_val_list[j] <= least_prob + args.delta]
                assert cand_list != []
                bf_length_list = [bf_list[label][j] for j in cand_list if bf_list[label][j] <= cur_pkt[1] + args.budget_per_pkt]
                
                # print("least_j_val: ", least_j_val)
                # print("bf_list[label][least_j_val]: ", bf_list[label][least_j_val])
                cur_pkt[1] = np.min(bf_length_list) if bf_length_list != [] else cur_pkt[1]
                # cur_pkt[1] = 1500
                adversarial_pkts.append(cur_pkt)
        res_data.append({
            "org_packets": flow["packets"],
            "adv_packets": adversarial_pkts,
            "source_file": flow["source_file"],
            "label": label
        })
        # print("res_data: ", res_data)
    return res_data
                
            
                
def generate_adv_actions(adversarial_data):
    adv_res = []
    pad_cnt_list = []
    pad_bytes_list = []
    pad_cnt_num = []
    insert_cnt = []
    for flow in adversarial_data:
        pad_cnt = 0
        pad_bytes = 0
        adv_actions = []
        # assert len(flow["org_packets"]) == len(flow["adv_packets"])
        inserted_num = 0
        for i in range(len(flow["adv_packets"])):
            org_pkt = flow["org_packets"][i - inserted_num]
            adv_pkt = flow["adv_packets"][i]
            if adv_pkt[3] == 1:
                inserted_num += 1
                assert inserted_num <= args.max_insert
                if np.random.rand() < 1:
                    adv_actions.append({
                        "action": "inserting",
                        "value": 1600,
                        "added_delay": 0
                    })
                else:
                    adv_actions.append({
                        "action": "inserting",
                        "value": 500,
                        "added_delay": 0.01
                    })
                continue
            adv_pkt = flow["adv_packets"][i]
            diff_in_length = adv_pkt[1] - org_pkt[1]
            assert diff_in_length >= 0
            rand_prob = np.random.rand()
            if diff_in_length == 0 or rand_prob > args.probability_threshold:
                adv_actions.append({
                        "action": "inaction",
                        "value": 0,
                        "added_delay": 0
                    })
            else:
                pad_cnt += 1
                pad_bytes += int(diff_in_length)
                adv_actions.append({
                        "action": "padding",
                        "value": int(diff_in_length) if int(diff_in_length) > 100 else int(100*(1 + np.random.rand())),
                        "added_delay": 0
                    })
        adv_res.append({
            "actions": adv_actions,
            "source_file": flow["source_file"]
        })
        pad_cnt_list.append(pad_cnt/len([pkt for pkt in flow["org_packets"] if pkt[0] == 1]))
        pad_cnt_num.append(pad_cnt)
        pad_bytes_list.append(pad_bytes)
        insert_cnt.append(inserted_num)
    # with open(f"statistic/{args.dataset}_{args.mode}_{args.target}_{args.basic_feature_num}_pad_cnt.json", "w") as f:
    #     json.dump(pad_cnt_list, f, ensure_ascii=False, indent=4)
    avg_pad_bytes = sum(pad_bytes_list) / len(pad_bytes_list)
    # print("avg_pad_bytes: ", avg_pad_bytes)
    # print("avg_pad_cnt: ", sum(pad_cnt_num) / len(pad_cnt_num))
    # print("avg_insert_cnt: ", sum(insert_cnt) / len(insert_cnt))
    return adv_res
            
            
def fit_attack_intensity(attack_data, args):
    """
    Fit the attack intensity based on the adversarial data.
    """
    base_insert_len = 1000
    base_delay = 0.05
        
    processed_flows = []
        
    for flow in attack_data:
        actions = flow['actions']
        new_actions = []
        action_budgets = [(action['value']/ args.attack_beta_length) * (args.attack_beta_length + args.attack_beta_time_ms) + (action['added_delay'] * 1000 / args.attack_beta_time_ms) * (args.attack_beta_length + args.attack_beta_time_ms) for action in actions if action['action'] == 'padding' or action['action'] == 'inaction']
        n = len([action for action in actions if action['action'] == 'padding' or action['action'] == 'inaction'])
        # print(n)
        d = int(n - int((1 - args.attack_pr_sel) * n)) + args.attack_insert_pkts
        # print(f"n: {n}, d: {d}")
        tops = torch.argsort(torch.tensor(action_budgets), descending = True)
        # print(tops)
        l2_norm_list = [torch.sum(torch.tensor(action_budgets)[tops[i:d+i]]) for i in range(n-d+1)]
        if len(l2_norm_list) < 1 or l2_norm_list[-1] >= args.attack_r_additive_star:
            choice = 0
            inserted_pkts = 0
            for action in actions:
                # assert action['added_delay'] == 0
                probbility = np.random.rand()
                if action['action'] == 'padding' or action['action'] == 'inaction':
                    new_actions.append({
                        'action': 'padding',
                        'value': int(args.attack_beta_length / (args.attack_beta_length + args.attack_beta_time_ms) * (args.attack_r_additive_star / d /3) * 2 * ( 1 - np.random.rand())),
                        'added_delay': float(args.attack_beta_time_ms / (args.attack_beta_length + args.attack_beta_time_ms) * (args.attack_r_additive_star / d /3) / 1000)* 2 * ( 1 - np.random.rand()) if np.random.rand() < 0.5 and args.attack_insert_pkts!=0 else 0
                        # 'added_delay': delay_max * np.random.rand() if np.random.rand() < delay_prob else 0
                    })
                elif action['action'] == 'inserting':
                    if inserted_pkts < args.attack_insert_pkts:
                        new_actions.append({
                            'action': 'inserting',
                            'value': int( base_insert_len ),
                            'added_delay': base_delay
                        })
                        inserted_pkts += 1
                    else:
                        continue
                else:
                    assert False
                    continue
                if np.random.rand() < 3 * args.attack_insert_pkts / n and inserted_pkts < args.attack_insert_pkts:
                    new_actions.append({
                            'action': 'inserting',
                            'value': int( base_insert_len ),
                            'added_delay': base_delay
                        })
                    inserted_pkts += 1
            if inserted_pkts < args.attack_insert_pkts:
                for i in range(args.attack_insert_pkts - inserted_pkts):
                    new_actions.append({
                        'action': 'inserting',
                        'value': int( base_insert_len),
                        'added_delay': base_delay
                    })
        else:
            choice = 1
            min_idx = min([i for i, val in enumerate(l2_norm_list) if val <= args.attack_r_additive_star])
            remaining_budget = args.attack_r_additive_star ** 2 - l2_norm_list[min_idx] - 1e-5
            remaining_budget = max(remaining_budget, 0)
            least_budget = action_budgets[tops[min_idx + d - 1]]
            # if least_budget == 0:
            #     print("least_budget is 0")
            #     assert False
            padding_idx = 0
            inserted_pkts = 0
            # print(int(beta_length * np.sqrt(least_budget)))
            for action in actions:
                if action['action'] == 'padding' or action['action'] == 'inaction':
                    if padding_idx in tops[:min_idx]:
                        new_actions.append({
                            'action': 'padding',
                            'value': action['value'],
                            'added_delay': action['added_delay']
                        })
                    elif padding_idx == tops[min_idx]:
                        new_actions.append({
                            'action': 'padding',
                            'value': int(args.attack_beta_length * (action_budgets[padding_idx] + remaining_budget/d) / (args.attack_beta_length + args.attack_beta_time_ms)) * 2 * ( 1 - np.random.rand()),
                            'added_delay': 0
                        })
                    else:
                        new_actions.append({
                            'action': 'padding',
                            'value': int(args.attack_beta_length * (action_budgets[padding_idx] + remaining_budget/d) / (args.attack_beta_length + args.attack_beta_time_ms)) * 2 * ( 1 - np.random.rand()),
                            'added_delay': 0
                        })
                    padding_idx += 1
                elif action['action'] == 'inserting' and inserted_pkts < args.attack_insert_pkts:
                    new_actions.append({
                        'action': 'inserting',
                        'value': int( base_insert_len ),
                        'added_delay': base_delay
                    })
                    inserted_pkts += 1
                else:
                    assert action['action'] == 'inserting'
                    continue
                if np.random.rand() < 3 * args.attack_insert_pkts / n and inserted_pkts < args.attack_insert_pkts:
                    new_actions.append({
                            'action': 'inserting',
                            'value': int( base_insert_len ),
                            'added_delay': base_delay
                        })
                    inserted_pkts += 1
            if inserted_pkts < args.attack_insert_pkts:
                for i in range(args.attack_insert_pkts - inserted_pkts):
                    new_actions.append({
                        'action': 'inserting',
                        'value': int( base_insert_len ),
                        'added_delay': base_delay
                    })
        new_n = len([action for action in new_actions if action['action'] == 'padding' or action['action'] == 'inaction'])
        assert new_n == n, f"new_n: {new_n}, n: {n}"
        validate_budget = torch.tensor([(action['value']/ args.attack_beta_length) **2 + (action['added_delay'] * 1000 / args.attack_beta_time_ms) **2 for action in new_actions if action['action'] == 'padding' or action['action'] == 'inaction'])
        tops = torch.argsort(validate_budget, descending = True)
        tops_d = tops[:d]
        # assert torch.sum(validate_budget[tops_d]) <= (l2_norm ** 2), f"l2_norm: {torch.sum(validate_budget[tops_d]) ** 0.5}, l2_norm_target: {l2_norm} , choice: {choice}"
        
        
        processed_flows.append({
            'actions': new_actions,
            'source_file': flow['source_file']
        })

    return processed_flows
                
                
            
            
                

if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="untargeted", choices=["targeted", "untargeted"])
    parser.add_argument("--target", type=int, default=0, help="Target class for targeted attack")
    # parser.add_argument("--config_path", type=str, default="config/config.json", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="CICDOH20", choices=["CICDOH20", "CICIOT2023", "TIISSRC23"])
    parser.add_argument("--basic_feature_num", type=int, default=10, help="Number of basic features")
    parser.add_argument("--max_padding_length", type=int, default=100, help="Maximum padding length")
    parser.add_argument("--long_feature_num", type=int, default=4, help="Number of long features")
    parser.add_argument("--max_flow_length", type=int, default=100, help="Maximum length of a flow")
    parser.add_argument("--bin_count", type=int, default=5, help="Number of bins for histogram")
    parser.add_argument("--max_insert", type=int, default=2, help="Maximum number of insertions")
    parser.add_argument("--enable_equal", type=bool, default=False, help="Whether to enable equal")
    parser.add_argument("--probability_threshold", type=float, default=0.9, help="Threshold for probability")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--budget_per_pkt", type=int, default=200, help="Budget per packet")
    parser.add_argument("--attack_beta_length", type=float, default=100)
    parser.add_argument("--attack_beta_time_ms", type=float, default=40)
    parser.add_argument("--attack_pr_sel", type=float, default=0.15)
    parser.add_argument("--attack_r_additive_star", type=float, default=21.958)
    parser.add_argument("--attack_insert_pkts", type=int, default=2)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    args.delta = 0.01
    # load config file
    # with open(args.config_path, "r") as f:
    #     config = json.load(f)
    #     args.train_set = config["dataset"][args.dataset]["train"]
    #     args.valid_set = config["dataset"][args.dataset]["valid"]
    #     args.test_set = config["dataset"][args.dataset]["test"]
    args.train_set = f"./dataset/{args.dataset}/json/train.json"
    args.valid_set = f"./dataset/{args.dataset}/json/valid.json"
    args.test_set = f"./dataset/{args.dataset}/json/test.json"
        
    # load and preprocess data
    args.train_data = load_data(args, args.train_set)
    args.valid_data = load_data(args, args.valid_set)
    args.test_data = load_data(args, args.test_set)
    
    # Temporal Feature Learning
    basic_features = temporal_feature_learning(args, args.train_data)
    args.label_num = len(basic_features.keys())
    print("get basic features done!")
    
    # classify and adjust the flows
    adjusted_train_data = classify_and_adjust_flows(args, basic_features, args.train_data)
    print("adjusted flows done!")
    
    # State Transition Modeling
    transition_model = state_transition_modeling(args, adjusted_train_data, basic_features)
    print("state transition modeling done!")
    
    # generate adversarial examples
    if args.mode == "targeted":
        adversarial_data = targeted_attack(args, basic_features, transition_model, args.test_data)
    elif args.mode == "untargeted":
        adversarial_data = untargeted_attack(args, basic_features, transition_model, args.test_data)
    else:
        raise ValueError("Invalid mode")
    print("generate adversarial examples done!")

    
    # generate adversarial actions and save adversarial examples
    adversarial_results = generate_adv_actions(adversarial_data)
    fitted_attack_data = fit_attack_intensity(adversarial_results, args)
    
    args.save_dir = f"./attack/Prism/{args.dataset}/"
    if os.path.exists(args.save_dir) is False:
        os.makedirs(args.save_dir)
    args.save_dir += 'Prism_beta_length_{}_beta_time_ms_{}_pr_sel_{}_r_additive_star_{}_insert_pkts_{}/'.format(
        args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_sel, args.attack_r_additive_star, args.attack_insert_pkts)
    if os.path.exists(args.save_dir) is False:
        os.makedirs(args.save_dir)
        
    with open(args.save_dir + f"attack.json", "w") as f:
        json.dump(fitted_attack_data, f, ensure_ascii=False, indent=1)