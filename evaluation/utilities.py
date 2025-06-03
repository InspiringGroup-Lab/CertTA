import copy
import json
import math
import torch
import pickle
import numpy as np
from scapy.all import IPOption
from argparse import Namespace


def load_hyperparam(args, config_path):
    with open(config_path, mode="r", encoding="utf-8") as f:
        param = json.load(f)

    args_dict = vars(args)
    args_dict.update(param)
    args = Namespace(**args_dict)

    return args

def load_or_initialize_parameters(model, param_path=None, map_location=None):
    if param_path is None:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
    else:
        print("Initialize with saved parameters.")
        model.load_state_dict(torch.load(param_path, map_location=map_location), strict=False)

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers

def adjust_learning_rate(optimizer, 
                         epoch_idx, epochs_num, warmup_epochs,  
                         init_learning_rate, min_learning_rate):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch_idx < warmup_epochs:
        lr = init_learning_rate * epoch_idx / warmup_epochs 
    else:
        lr = min_learning_rate + (init_learning_rate - min_learning_rate) * 0.5 * \
            (1. + math.cos(math.pi * (epoch_idx - warmup_epochs) / (epochs_num - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def save_model(model_name, model, model_path):
    if model_name in ['Whisper', 'kFP', 'KMeans', 'Kitsune']:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        return
    
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

def logging(filename, line):
    with open(filename, 'a') as fp:
        fp.writelines(line)

def model_name_generator(args):
    model_name = args.model
    if args.augment == 'CertTA':
        model_name += '_CertTA_beta_length_{}_beta_time_ms_{}_pr_sel_{}'.format(
            args.beta_length, args.beta_time_ms, args.pr_sel)
    elif args.augment == 'VRS':
        model_name += '_VRS_sigma_{}'.format(args.sigma_vrs)
    elif args.augment == 'BARS':
        model_name += '_BARS'
    return model_name

def attack_name_generator(args):
    attack_name = args.attack
    if args.attack == 'Blanket':
        attack_name += '_beta_length_{}_beta_time_ms_{}_pr_sel_{}_r_additive_star_{}_insert_pkts_{}/'.format(
        args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_sel, args.attack_r_additive_star, args.attack_insert_pkts)
    # elif args.attack == 'Amoeba':
    #     if args.is_exp_1:
    #         attack_name += '_exp1_{}_{}_{}_{}_{}'.format(args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_del, args.attack_R_additive_l2, args.attack_insert_pkts)
    #     else:
    #         attack_name += '_{}_{}_{}_{}_{}_{}'.format(args.Truncate_Penalty, args.Time_Penalty, args.Data_Penalty, args.padding_per_pkt, args.delay_per_pkt, args.insert_pkts)
    # elif args.attack == 'Prism':
    #     if args.is_exp_1:
    #         attack_name += '_exp1_{}_{}_{}_{}_{}'.format(args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_del, args.attack_R_additive_l2, args.attack_insert_pkts)
    #     else:
    #         attack_name += '_exp2_{}_{}_{}_{}'.format(args.Prism_BF_Num, args.attack_insert_pkts, args.Prism_prob, args.attack_R_additive_l2)
    return attack_name

def print_error(value):
        print("error: ", value)

def macro_acc_from_confuse_matrix(conf_mat):
    labels_num = conf_mat.shape[0]
    acc_label = []
    for true_label in range(labels_num):
        acc_label.append(conf_mat[true_label][true_label] / sum(conf_mat[true_label]))
    macro_acc = sum(acc_label) / labels_num
    return macro_acc

def metric_from_confuse_matrix(conf_mat):
    r"""Calc f1 score from true_label list and pred_label list
    Args:
        conf_mat (np.array): 2D confuse_matrix of int, conf_mat[i,j]
            stands for samples with true label_i predicted as label_j.
    Returns:
        micro_p_r_f (float): micro precision and recall
        macro_f1 (float): macro_f1 value
        macro_p (float): macro_precision value
        macro_r (float): macro_recall value
    """
    def clac_f1(precisions, recalls):
        f1s = []
        epsilon = 1e-6
        for (p,r) in zip(precisions, recalls):
            if (p>epsilon) and (r>epsilon):
                f1s.append( (2*p*r) / (p+r) )
            else:
                f1s.append(0.0)
        return f1s

    def seperated_p_r(conf_mat):
        preds = conf_mat.sum(axis=0) * 1.0
        trues = conf_mat.sum(axis=1) * 1.0
        p, r = [], []
        for i in range(conf_mat.shape[0]):
            if preds[i]:
                p.append(conf_mat[i,i] / preds[i])
            else:
                p.append(0.0)
            r.append(conf_mat[i,i] / trues[i])
        return p, r
    
    precisions, recalls = seperated_p_r(conf_mat)
    f1s = clac_f1(precisions, recalls)

    return precisions, recalls, f1s

def ts2IPOption(ts):
    op_type_str = '01111111' # copy_flag = 0, optclass = 3, option = 31
    op_length_str = '00001000' # 8 Bytes = 64 bits 
    ts_us = int(ts * 1e6)
    b = bin(ts_us)
    op_value_str = ('0' * 48 + bin(ts_us)[2:])[-48:]
    op_str = op_type_str + op_length_str + op_value_str
    op_bytes = bytes([int(op_str[i:i+8], 2) for i in range(0, len(op_str), 8)])
    return IPOption(op_bytes)

def rewrite_pcap_packet_timestamp(pkt, eth_ts=None, ip_ts=None):
    if eth_ts is not None:
        pkt.time = eth_ts
    if ip_ts is not None:
        # try:
        ip_ts_option = pkt['IP'].options[0]
        assert ip_ts_option.copy_flag == 0
        assert ip_ts_option.optclass == 3
        assert ip_ts_option.option == 31
        
        rewrite_ip_ts_option = ts2IPOption(ip_ts)
        pkt['IP'].options[0] = rewrite_ip_ts_option
        assert pkt['IP'].ihl * 4 == len(pkt['IP']) - len(pkt['IP'].payload)
        # except:
        #     raise Exception('No IP Option for timestamp!')
    return pkt

def re_compute_chksum_wirelen(pkt):
    pkt.wirelen = None
    pkt['IP'].chksum = None # re-compute IP checksum
    pkt['TCP'].chksum = None # re-compute TCP checksum
    return pkt

def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    try:
        remanent_count = len(result[0])%4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def bigram_generation(packet_datagram, token_len = 64, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram,1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > token_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    
    return result

def min_np(a,b):
    return_x = copy.deepcopy(a)
    for i in range(len(return_x)):
        return_x[i] = min(return_x[i],b[i])
    return return_x

def max_np(a,b):
    return_x = copy.deepcopy(a)
    for i in range(len(return_x)):
        return_x[i] = max(return_x[i],b[i])
    return return_x

def get_x_norm(flows, packet_dim=3):
    min_a = np.zeros(packet_dim)
    max_a = np.zeros(packet_dim)
    for flow in flows:
        direction_length = flow['direction_length']
        timestamp = flow['timestamp']
        iat = [0] + [timestamp[i] - timestamp[i-1] for i in range(1, flow['packet_num'])]
        
        min_a = min_np(min_a, [min(iat), min(direction_length), min(timestamp)])
        max_a = max_np(max_a, [max(iat), max(direction_length), max(timestamp)])
    
    return [min_a, max_a]

class Normalizer:
    def __init__(self):
        super(Normalizer, self).__init__()
        
        self.norm_max = None
        self.norm_min = None

        self.eps = 1e-16
    
    def normalize(self, x):
        if self.norm_max is None or self.norm_min is None:
            self.norm_max = np.max(x, axis=0)
            self.norm_min = np.min(x, axis=0)
        return (x - self.norm_min) / (self.norm_max - self.norm_min + self.eps)

    def update(self, x):
        if self.norm_max is None or self.norm_min is None:
            self.norm_max = np.max(x, axis=0)
            self.norm_min = np.min(x, axis=0)
        else:
            self.norm_max = np.max(np.concatenate([x, self.norm_max.reshape(1, -1)], axis=0), axis=0)
            self.norm_min = np.min(np.concatenate([x, self.norm_min.reshape(1, -1)], axis=0), axis=0)