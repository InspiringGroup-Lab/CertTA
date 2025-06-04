import json
import random
import os
import argparse
import torch
import numpy as np

# Handle command line arguments
parser = argparse.ArgumentParser(description="Generate Amoeba attack actions")
parser.add_argument("--dataset", type=str, default="CICDOH20", help="Dataset to use (CICDOH20 or TIISSRC23)")
parser.add_argument("--attack_beta_length", type=float, default=100)
parser.add_argument("--attack_beta_time_ms", type=float, default=40)
parser.add_argument("--attack_pr_sel", type=float, default=0.15)
parser.add_argument("--attack_r_additive_star", type=float, default=21.958)
parser.add_argument("--attack_insert_pkts", type=int, default=2)
args = parser.parse_args()
args.attack = 'Amoeba'
args.save_dir = './{}/'.format(args.dataset)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
args.save_dir += 'Amoeba_beta_length_{}_beta_time_ms_{}_pr_sel_{}_r_additive_star_{}_insert_pkts_{}/'.format(
    args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_sel, args.attack_r_additive_star, args.attack_insert_pkts)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

random.seed(7)
dataset = args.dataset
root_dir = "./logs/Amoeba_raw_{}".format(dataset)


pad_buget_list = [150]
delay_buget_list = [50]
delay_buget_list = [x/1000 for x in delay_buget_list]
insertion_buget_list = [3]

with open(root_dir + "/records/test.json", "r") as f:
    atk_list = json.load(f)
    
with open("../../dataset/CICDOH20/json/test.json", "r") as f:
    ci20list = json.load(f)
    
with open("../../dataset/TIISSRC23/json/test.json", "r") as f: 
    ci23list = json.load(f)

# transform the attack actions
datalist = ci20list if dataset == "CICDOH20" else ci23list
for i in range(len(pad_buget_list)):
    pad_buget = pad_buget_list[i]
    delay_buget = delay_buget_list[i]
    insertion_buget = insertion_buget_list[i]
    result = []
    statistics = []
    for j in range(len(atk_list)):
        atk = atk_list[j]
        src_file = datalist[j]["source_file"]
        org_flow_len = datalist[j]["packet_num"]
        actions = []
        padding_bytes = 0
        added_delay = 0
        inserting_num = 0
        for action in atk["actions"]:
            if action["action"] == "padding":
                org_flow_len -= 1
                if action["value"] > 3 * pad_buget:
                    pad_val = pad_buget
                elif action["value"] < pad_buget / 2:
                    pad_val = random.randint(pad_buget / 2, pad_buget)
                else:
                    pad_val = random.randint(0, pad_buget) if action["value"] > pad_buget else action["value"]
                if action["added_delay"] == 0:
                    delay_val = random.randint(0, int(500*delay_buget))/1000
                else:
                    delay_val = random.randint(0, int(1000*delay_buget))/1000 if action["added_delay"] > delay_buget else action["added_delay"]
                padding_bytes += pad_val
                added_delay += delay_val
                actions.append({
                    "action": "padding",
                    "value": pad_val,
                    "added_delay": delay_val
                })
            elif action["action"] == "inserting":
                if inserting_num < insertion_buget:
                    inserting_num += 1
                    actions.append(action)
            else:
                raise Exception("Unknown action")
        for p in range(org_flow_len):
            actions.append({
                "action": "inaction",
                "value": 0,
                "added_delay": 0
            })
        result.append({
            "source_file": src_file,
            "actions": actions
        })
        statistics.append({
            "padding_bytes": padding_bytes,
            "added_delay": added_delay,
            "inserting_num": inserting_num
        })
    
    
# # fit the result to the intensity of the attack
# beta_length = 50
# beta_time = 10
# pr_del = 0.8
# max_insert_pkts_list = [1, 1, 0]
# l2_norm = args.attack_r_additive_star
# # max_insert_pkts = 0
# delay_max = 0.002
base_insert_len = 1000
base_delay = 0.05
# prob = -0.6 + l2_norm/20
# bias = 1.1 + l2_norm/20
# delay_prob = 0.8

# max_insert_pkts = max_insert_pkts_list[c_idx]
# print(f"max_insert_pkts: {max_insert_pkts}, l2_norm: {l2_norm}")

attack_data = result
    
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
    
    processed_flows.append({
        'actions': new_actions,
        'source_file': flow['source_file']
    })

with open(args.save_dir + "attack.json", "w") as f:
    json.dump(processed_flows, f, indent=4)