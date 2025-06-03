import os
import sys
import copy
import json
import binascii
import numpy as np
from tqdm import tqdm
from PIL import Image
import scapy.all as scapy
from decimal import Decimal
from torchvision import transforms
from evaluation.utilities import ts2IPOption, rewrite_pcap_packet_timestamp, re_compute_chksum_wirelen

sys.path.append('./model/TrafficFormer/data_generation')
from model.TrafficFormer.data_generation.finetuning_data_gen import *

sys.path.append('./model/kFP')
from model.kFP.feature_extractor import TOTAL_FEATURES # type: ignore


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


def flow_preprocessing(flow, args):
    assert flow['packet_num'] == len(flow['timestamp'])
    assert len(flow['timestamp']) == len(flow['direction_length'])
    assert flow['timestamp'][0] == 0

    if args.model == 'TrafficFormer':
        packets = flow['packets']
        packets = [pkt['IP'] for pkt in packets]
        start_index = args.start_index
        select_packet_len = args.payload_length
        packets_num = args.payload_packet
        add_sep = args.add_sep

        No_ether = False
        if not hasattr(packets[0],'type'): #no ether header
            No_ether = True
        if (not No_ether and packets[0].type == 0x86dd) or (No_ether and packets[0].version == 6): #do not handle IPV6
            raise Exception('Do not handle IPV6!')
        
        # packets = random_ip_port(packets)
        packets = random_tls_randomtime(packets)

        flow_data_string = ''
        packet_byte_arrays = []
        for packet in packets[:packets_num]:
            packet_data = packet.copy()
            
            data = (binascii.hexlify(bytes(packet_data)))
                    
            if No_ether:
                packet_string = data.decode()
                packet_string = packet_string[start_index-28:start_index-28+2*select_packet_len]
            else:
                raise Exception('!!!!!!!!!!')
                packet_string = data.decode()[start_index:start_index+2*select_packet_len]
            
            if len(packet_string) < 2 * select_packet_len:
                packet_string += '0' * (2 * select_packet_len - len(packet_string))
            
            if add_sep:
                flow_data_string += "[SEP] "
            
            flow_data_string += bigram_generation(packet_string.strip(), token_len=len(packet_string.strip()), flag = True)
            packet_byte_array = np.array([int(packet_string.strip()[i:i + 2], 16) for i in range(0, len(packet_string.strip()), 2)], dtype=np.int8)
            packet_byte_arrays.append(packet_byte_array)
            
        src = args.tokenizer.convert_tokens_to_ids(["[CLS]"] + args.tokenizer.tokenize(flow_data_string))
        seg = [0] * len(src)
        tgt = flow['label']
        if len(src) > args.seq_length:
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)

        return (src, tgt, seg, packet_byte_arrays)
    elif args.model == 'YaTC':
        header_hexes = 2 * args.header_bytes
        payload_hexes = 2 * args.payload_bytes
        packets_num = args.input_packets

        headers_payloads = []
        packets = flow['packets']
        for packet in packets[:packets_num]:
            header_payload = (binascii.hexlify(bytes(packet['IP']))).decode()
            payload = (binascii.hexlify(bytes(packet['TCP'].payload))).decode()
            header = header_payload.replace(payload, '')
            
            if len(header) > header_hexes:
                header = header[:header_hexes]
            elif len(header) < header_hexes:
                header += '0' * (header_hexes - len(header))

            if len(payload) > payload_hexes:
                payload = payload[:payload_hexes]
            elif len(payload) < payload_hexes:
                payload += '0' * (payload_hexes - len(payload))
            
            headers_payloads.append(header + payload)
        
        if len(headers_payloads) < packets_num:
            for _ in range(packets_num - len(headers_payloads)):
                headers_payloads.append('0' * (header_hexes + payload_hexes))
        flow_hex_string = ''.join(headers_payloads)

        flow_byte_array = np.array([int(flow_hex_string[i:i + 2], 16) for i in range(0, len(flow_hex_string), 2)])
        flow_byte_array = np.reshape(flow_byte_array, (-1, args.row_bytes))
        flow_byte_array = np.uint8(flow_byte_array)
        flow_img = Image.fromarray(flow_byte_array)

        mean = [0.5]
        std = [0.5]
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        src_img = transform(flow_img)
        tgt = flow['label']

        return (src_img, tgt, flow_byte_array)
    elif args.model == 'DF':
        direction_length = copy.deepcopy(flow['direction_length'])
        timestamp_ms = [i * 1e3 for i in flow['timestamp']]
        iat_ms = [0] + [timestamp_ms[i] - timestamp_ms[i-1] for i in range(1, flow['packet_num'])]
        packet_num = flow['packet_num']
        LENGTH = args.max_flow_length
        if packet_num < LENGTH:
            direction_length += [0] * (LENGTH - packet_num)
            iat_ms += [0] * (LENGTH - packet_num)
        else:
            direction_length = direction_length[:LENGTH]
            iat_ms = iat_ms[:LENGTH]
        src_seq = [direction_length, iat_ms]
        tgt = flow['label']

        return (src_seq, tgt)
    elif args.model == 'Whisper':
        direction_length = flow['direction_length']
        timestamp_ms = [i * 1e3 for i in flow['timestamp']]
        iat_ms = [0] + [timestamp_ms[i] - timestamp_ms[i-1] for i in range(1, flow['packet_num'])]
        packet_num = flow['packet_num']
        src_S = [[direction_length[i], iat_ms[i]] for i in range(packet_num)]
        tgt = flow['label']

        return (src_S, tgt)
    elif args.model in ['Kitsune', 'kFP', 'KMeans']:
        direction_length = flow['direction_length']
        timestamp_ms = [i * 1e3 for i in flow['timestamp']]
        packet_num = flow['packet_num']
        cell = [[timestamp_ms[i], direction_length[i]] for i in range(packet_num)]
        src_feat = TOTAL_FEATURES(cell, max_size=args.feature_num)
        tgt = flow['label']
        if args.normalizer is not None:
            src_feat = args.normalizer.normalize(np.array(src_feat)).tolist()

        return (src_feat, tgt)


def dimension_alignment(args, instance_flow):
    if args.model == 'Whisper':
        src_S, tgt = instance_flow
        for _ in range(len(src_S), args.max_flow_length):
            src_S.append([0, 0])
        instance_flow = (src_S, tgt)

    return instance_flow


def load_data(data_file, pcap_level=False, max_flow_length=100, truncate=None):
    flows = []
    with open(data_file) as fp:
        data = json.load(fp)
    for flow in data:
        pcap_file = './dataset/' + flow['source_file'].split('./')[-1]
        flow['source_file'] = pcap_file
        if truncate is not None:
            packet_num = int(flow['packet_num'] * truncate)
            flow['packet_num'] = packet_num
            flow['direction_length'] = flow['direction_length'][:packet_num]
            flow['timestamp'] = flow['timestamp'][:packet_num]

    if pcap_level is True:
        for flow in tqdm(data, desc='Loading pcaps'):
            pcap_file = flow['source_file']

            pkts = scapy.rdpcap(pcap_file, count=min(max_flow_length, flow['packet_num']))
            
            prev_src_sport = ''
            direction = -1
            for idx, pkt in enumerate(pkts):
                # Mask IP addresses & ports 
                ip = pkt['IP'].copy()
                tcp = pkt['TCP'].copy()
                header_payload = (binascii.hexlify(bytes(pkt['IP']))).decode()
                
                src_sport = ip.src + '-' + str(tcp.sport)
                if src_sport != prev_src_sport:
                    direction = - direction
                    prev_src_sport = src_sport

                ip.payload = scapy.NoPayload()
                ip_hex_string = (binascii.hexlify(bytes(ip))).decode()
                tcp.payload = scapy.NoPayload()
                tcp_hex_string = (binascii.hexlify(bytes(tcp))).decode()
                if direction == 1:
                    header_payload = header_payload.replace(ip_hex_string[:40], ip_hex_string[:24] + 'ffffffff00000000')
                    header_payload = header_payload.replace(tcp_hex_string, 'ffff0000' + tcp_hex_string[8:])
                else:
                    header_payload = header_payload.replace(ip_hex_string[:40], ip_hex_string[:24] + '00000000ffffffff')
                    header_payload = header_payload.replace(tcp_hex_string, '0000ffff' + tcp_hex_string[8:])

                pkt['IP'] = scapy.IP(binascii.unhexlify(header_payload))

                # Insert timestamp in IP options 
                ip_ts = flow['timestamp'][idx]
                ip_ts_option = ts2IPOption(ip_ts)
                pkt['IP'].options.insert(0, ip_ts_option)
                ihl_byte = len(pkt['IP']) - len(pkt['IP'].payload)
                if ihl_byte % 4 != 0:
                    raise Exception('Wrong IP options length!')
                pkt['IP'].ihl = ihl_byte // 4
                pkt['IP'].len = len(pkt['IP'])
                
                re_compute_chksum_wirelen(pkt)
            
            tmp_pcap_file = './evaluation/tmp/loading_flow_{}.pcap'.format(os.getpid())
            scapy.wrpcap(tmp_pcap_file, pkts)
            flow['packets'] = scapy.rdpcap(tmp_pcap_file)
            os.remove(tmp_pcap_file)

            flows.append(flow)
    else:
        flows = data
    return flows


def generate_attack_flow(args, flow, attack_actions):
    attack_flow = {
            'direction_length': [],
            'timestamp': [],

            'packet_num': None,
            'label': flow['label'],
            'source_file': flow['source_file']
        }

    org_direction_length = flow['direction_length']
    org_timestamp = flow['timestamp']
    org_iat = [0] + [org_timestamp[i] - org_timestamp[i-1] for i in range(1, len(org_timestamp))]
    org_packet_num = len(flow['direction_length'])
    if args.pcap_level:
        attack_flow['packets'] = []
        org_packets = flow['packets']
        pcap_ts_start = org_packets[0].time

    # inserting in the flow start for TrafficFormer & YaTC
    if args.model in ['TrafficFormer', 'YaTC']:
        new_attack_actions = []
        for action in attack_actions:
            if action['action'] == 'inserting':
                new_attack_actions.insert(0, action)
            else:
                new_attack_actions.append(action)
        attack_actions = new_attack_actions
    
    # Apply attack action to each packet 
    org_pkt_idx = 0
    current_ts = 0
    for action_idx, action in enumerate(attack_actions):
        if action['action'] == 'padding':
            if len(attack_flow['timestamp']) > 0:
                current_ts += org_iat[org_pkt_idx] + action['added_delay']
            attack_flow['timestamp'].append(current_ts)
            pad_length = action['value']
            direction = np.sign(org_direction_length[org_pkt_idx])
            attack_flow['direction_length'].append(org_direction_length[org_pkt_idx] + direction * pad_length)
            if args.pcap_level:
                pkt = org_packets[org_pkt_idx]
                pkt = rewrite_pcap_packet_timestamp(pkt, 
                        eth_ts=pcap_ts_start + Decimal(current_ts),
                        ip_ts=current_ts) # timestamp
                if pad_length > 0: # packet length & payload
                    pkt['IP'].len = pkt['IP'].len + pad_length
                    pkt['TCP'].add_payload(bytes([random.randint(0, 255) for _ in range(pad_length)]))
                attack_flow['packets'].append(re_compute_chksum_wirelen(pkt))
            org_pkt_idx += 1
        elif action['action'] == 'inserting':
            if len(attack_flow['timestamp']) > 0:
                current_ts += action['added_delay']
            attack_flow['timestamp'].append(current_ts)
            pad_length = action['value']
            direction = np.sign(org_direction_length[org_pkt_idx - 1 if org_pkt_idx >= org_packet_num else org_pkt_idx])
            attack_flow['direction_length'].append(direction * pad_length)
            if args.pcap_level:
                pkt = org_packets[org_pkt_idx - 1 if org_pkt_idx >= org_packet_num else org_pkt_idx].copy()    
                # timestamp
                pkt.time = pcap_ts_start + Decimal(current_ts)
                pkt['IP'].options = [ts2IPOption(current_ts)]
                pkt['IP'].ihl = (len(pkt['IP']) - len(pkt['IP'].payload)) // 4
                # packet length & payload
                pkt['TCP'].options = []
                pkt['TCP'].payload = scapy.NoPayload()
                pkt['TCP'].dataofs = len(pkt['TCP']) // 4
                pkt['TCP'].add_payload(bytes([random.randint(0, 255) for _ in range(pad_length - 40)]))
                pkt['IP'].len = len(pkt['IP'])
                assert pkt['IP'].len == pad_length + 8
                attack_flow['packets'].append(re_compute_chksum_wirelen(pkt))
        elif action['action'] == 'inaction':
            current_ts += org_iat[org_pkt_idx]
            attack_flow['timestamp'].append(current_ts)
            attack_flow['direction_length'].append(org_direction_length[org_pkt_idx])
            if args.pcap_level:
                pkt = rewrite_pcap_packet_timestamp(org_packets[org_pkt_idx],
                        eth_ts=pcap_ts_start + Decimal(current_ts),
                        ip_ts=current_ts) # timestamp
                attack_flow['packets'].append(re_compute_chksum_wirelen(pkt))
            org_pkt_idx += 1
        else:
            raise Exception('Unknown attack action: {}!'.format(action['action']))

    # Truncate the flow according to args.max_flow_length
    attack_flow['direction_length'] = attack_flow['direction_length'][:args.max_flow_length]
    attack_flow['packet_num'] = len(attack_flow['direction_length'])
    attack_flow['timestamp'] = attack_flow['timestamp'][:args.max_flow_length]
    if args.pcap_level:
        attack_flow['packets'] = attack_flow['packets'][:args.max_flow_length]
        tmp_pcap_file = './evaluation/tmp/attack_flow_{}.pcap'.format(os.getpid())
        scapy.wrpcap(tmp_pcap_file, attack_flow['packets'])
        attack_flow['packets'] = scapy.rdpcap(tmp_pcap_file)
        os.remove(tmp_pcap_file)
    
    return attack_flow


def generate_attack_flows(args, test_flows):
    attack_set_dict = {}
    with open(args.attack_set_path) as fp:
        attack_set = json.load(fp)
    for sample in attack_set:
        source_file = './dataset/' + sample['source_file'].split('./')[-1]
        attack_set_dict[source_file] = sample['actions']

    attack_flows = []
    for flow in tqdm(test_flows, desc='Generating attack flows'):
        attack_flow = generate_attack_flow(args, flow, attack_set_dict[flow['source_file']])
        attack_flows.append(attack_flow)

    return attack_flows
