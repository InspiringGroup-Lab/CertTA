import os
import torch
import random
import binascii
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import vmap
import scapy.all as scapy
from torchvision import transforms
from evaluation.utilities import rewrite_pcap_packet_timestamp, re_compute_chksum_wirelen, bigram_generation


'''
    input flow format:
        {
            // sequential attributes
            'direction_length': [int, ...],
            'timestamp': [float, ...],
            
            // scalar attributes
            'packet_num': int,
            'label': int,
            'source_file': str,
            
            // optional, for pcap level perturbations, returned object of scapy.rdpcap()
            'packets': [packet, ...], 
        }
'''


def smoothing_joint(flow, smoothing_params, pcap_level=False):
    '''
        smoothing_params = {
            // packet timing delay
            'beta_time_ms': int or None,
            // pcaket length padding
            'beta_length': int or None,
            // packet selection
            'pr_sel': float or None
        }
    '''
    if pcap_level is True:
        assert 'packets' in flow.keys()
        assert len(flow['packets']) == flow['packet_num']

    # packet selection
    packet_num = flow['packet_num']
    sel_pos = list(range(packet_num))
    pr_sel = smoothing_params['pr_sel']
    if pr_sel is not None:
        assert pr_sel > 0 and pr_sel <= 1
        d_sel = int(np.ceil(pr_sel * packet_num))
        sel_pos = random.sample(list(range(packet_num)), d_sel)
    new_packet_num = len(sel_pos)
    
    # packet timing delay
    timestamp = flow['timestamp']
    iat = [0] + [timestamp[i] - timestamp[i-1] for i in range(1, flow['packet_num'])]
    new_iat = [iat[pos] for pos in sel_pos]
    beta_time_ms = smoothing_params['beta_time_ms']
    if beta_time_ms is not None:
        assert beta_time_ms > 0
        new_iat = [t + np.random.exponential(beta_time_ms) * 0.001 for t in new_iat]
    new_timestamp = [0]
    for t in new_iat[1:]:
        new_timestamp.append(new_timestamp[-1] + float(t) * np.sqrt(2 / np.pi))
        
    # packet length padding
    pad = [0] * new_packet_num
    beta_length = smoothing_params['beta_length']
    if beta_length is not None:
        assert beta_length > 0
        pad = [int(np.random.exponential(beta_length)) for _ in range(new_packet_num)]
    direction_length = flow['direction_length']
    new_direction_length = [direction_length[pos] for pos in sel_pos]
    new_direction_length = [(length + np.sign(length) * int(pad[i] * np.sqrt(2 / np.pi))) 
                                for i, length in enumerate(new_direction_length)]
    # if the first packet is deleted, reset the direction of the new first packet as +1
    if new_direction_length[0] < 0:
        new_direction_length = [-length for length in new_direction_length]
    
    ########################################################################
    # generate the perturbated flow
    new_flow = {
        'direction_length': new_direction_length,
        'timestamp': new_timestamp,
        'packet_num': new_packet_num,
        'label': flow['label'],
        'source_file': flow['source_file']
    }
    
    # update packet bytes
    if pcap_level is True:
        packets = [flow['packets'][pos] for pos in sel_pos]

        for i in range(new_packet_num):
            pkt = packets[i]
            # timestamp
            pkt = rewrite_pcap_packet_timestamp(pkt, ip_ts=new_flow['timestamp'][i])

            # re-compute checksum & wirelen
            pkt = re_compute_chksum_wirelen(pkt)

            # packet length & payload
            pad_length = pad[i]
            if pad_length > 0:
                pkt['IP'].len = pkt['IP'].len + pad_length
                pkt['TCP'].add_payload(bytes([random.randint(0, 255) for _ in range(pad_length)]))
        
        tmp_pcap_file = './evaluation/tmp/perturbated_flow_{}.pcap'.format(os.getpid())
        scapy.wrpcap(tmp_pcap_file, packets)
        new_flow['packets'] = scapy.rdpcap(tmp_pcap_file)
        os.remove(tmp_pcap_file)

    return new_flow


def generate_smoothing_samples(flow, smoothing_params, samples_num, pcap_level=False):
    smoothing_samples = []    
    for _ in tqdm(range(samples_num), desc='Generating smoothing samples (CertTA)', disable=True):
        sample = smoothing_joint(flow, smoothing_params, pcap_level)
        smoothing_samples.append(sample)
    return smoothing_samples


def smoothing_vrs(instance_flow, sigma_vrs, args):
    if args.model == 'TrafficFormer':
        _, tgt, _, packet_byte_arrays = instance_flow
        for i in range(len(packet_byte_arrays)):
            noise = np.random.normal(0.0, sigma_vrs, packet_byte_arrays[i].shape)
            noise = np.round(noise).astype(np.int8)
            packet_byte_arrays[i] += noise
            for v in np.nditer(packet_byte_arrays[i], op_flags=['readwrite']):
                v[...] = np.int8(int8_overflow(v))
        
        flow_data_string = ''
        for packet_byte_array in packet_byte_arrays:
            packet_string = ''
            for byte in packet_byte_array:
                packet_string += (binascii.hexlify(byte)).decode()
            if args.add_sep:
                flow_data_string += "[SEP] "
            flow_data_string += bigram_generation(packet_string.strip(), token_len=len(packet_string.strip()), flag = True)
        
        src = args.tokenizer.convert_tokens_to_ids(["[CLS]"] + args.tokenizer.tokenize(flow_data_string))
        seg = [0] * len(src)
        if len(src) > args.seq_length:
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)
            
        instance_flow = (src, tgt, seg, packet_byte_arrays)
    elif args.model == 'YaTC':
        _, tgt, flow_byte_array = instance_flow
        x = flow_byte_array
        noise = np.random.normal(0.0, sigma_vrs, x.shape)
        noise = np.round(noise)
        x_noised = x + noise
        for v in np.nditer(x_noised, op_flags=['readwrite']):
            v[...] = int8_overflow(v, 255)

        flow_img = Image.fromarray(x_noised)
        mean = [0.5]
        std = [0.5]
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        src_img = transform(flow_img)

        instance_flow = (src_img, tgt, x_noised)
    else:
        src, tgt = instance_flow
        x = np.array(src).astype(np.float32)
        noise = np.random.normal(0.0, sigma_vrs, x.shape).astype(np.float32)
        x_noised = x + noise
        instance_flow = (x_noised, tgt)

    return instance_flow


def generate_smoothing_samples_vrs(instance_flow, sigma_vrs, samples_num, args):
    smoothing_samples = []
    for _ in tqdm(range(samples_num), desc='Generating smoothing samples (VRS)', disable=True):
        sample = smoothing_vrs(instance_flow, sigma_vrs, args)
        smoothing_samples.append(sample)
    return smoothing_samples


def smoothing_bars(instance_flow, noise_generator, t, model_name):
    if model_name == 'TrafficFormer':
        src, tgt, seg = instance_flow
        X = torch.LongTensor(src)
    elif model_name == 'YaTC':
        _, tgt, flow_byte_array = instance_flow
        X = torch.tensor(flow_byte_array)
    elif model_name in ['DF', 'Kitsune']:
        src, tgt = instance_flow
        X = torch.tensor(src)
    else:
        raise Exception('BARS is not applicable to model', model_name)
    X = X.to(noise_generator.device)

    noise_feat = noise_generator.sample_feat(1) * t
    noise_feat = noise_feat.reshape(X.shape)
    noised_X = X + noise_feat

    if model_name == 'TrafficFormer':
        noised_X = vmap(vocab_overflow)(noised_X)
        noised_X = noised_X.detach().cpu().numpy().tolist()
        instance_flow = (noised_X, tgt, seg)
    elif model_name == 'YaTC':
        noised_X = vmap(int8_overflow)(noised_X)
        noised_X = noised_X.unsqueeze(0)
        noised_X = torch.clamp(noised_X, 0, 255)
        mean = [0.5]
        std = [0.5]
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean, std),
        ])
        src_img = transform(noised_X)
        noised_X = noised_X.long().squeeze(0).detach().cpu().numpy().tolist()
        instance_flow = (src_img, tgt, noised_X)
    elif model_name in ['DF', 'Kitsune']:
        noised_X = noised_X.detach().cpu().numpy().tolist()
        instance_flow = (noised_X, tgt)
    
    return instance_flow

def generate_smoothing_samples_bars(instance_flow, noise_generator, t, model_name, samples_num):
    smoothing_samples = []
    for _ in tqdm(range(samples_num), desc='Generating smoothing samples (BARS)', disable=True):
        sample = smoothing_bars(instance_flow, noise_generator, t, model_name)
        smoothing_samples.append(sample)
    return smoothing_samples


def int8_overflow(x):
    while x > 255:
        x -= 255    
    while x < 0:
        x += 255
    return x


def vocab_overflow(x):
    vocab = 60005 - 1
    while x > vocab:
        x -= vocab    
    while x < 0:
        x += vocab
    return x