import os
import sys
import time
import json
import random
import argparse
from tqdm import tqdm
import scapy.all as scapy
from pathos.multiprocessing import ProcessingPool as Pool

os.environ['CUDA_VISIBLE_DEVICES']='2'
sys.path.append('.') # run in directory: CertTA_public/
from evaluation.utilities import *
from evaluation.opts import smoothing_opts, attack_opts, training_opts
from evaluation.data_loader import load_data, generate_attack_flows, flow_preprocessing, dimension_alignment
from certification.smoothing import smoothing_vrs, generate_smoothing_samples_bars
from BARS.train_noise_generator import create_noise_generator

sys.path.append('./model/kFP')
from model.kFP.model_kFP import kFP

sys.path.append('./model/Kitsune')
from model.Kitsune.KitNET import KitNET

sys.path.append('./model/Whisper')
from model.Whisper.model_Whisper import Whisper

sys.path.append('./model/DF')
from model.DF.DFNet import DFNet

sys.path.append('./model/YaTC')
import model.YaTC.models_YaTC as models_YaTC

sys.path.append('./model/TrafficFormer')
from model.TrafficFormer.fine_tuning.run_classifier import Classifier
from model.TrafficFormer.uer.utils import *


def initialize_model(args):
    if args.model == 'kFP':
        model = kFP(args)
        with open(args.save_dir + 'best_model.bin', 'rb') as file:
            model = pickle.load(file)
            
    elif args.model == 'Kitsune':
        args.d = args.feature_num

        model = KitNET(args.feature_num, args.labels_num, args.maxAE, feature_map=[[i for i in range(args.feature_num)]], device=args.device)
        with open(args.save_dir + 'best_model.bin', 'rb') as file:
            model = pickle.load(file)
        model = model.to(args.device)
        
    elif args.model == 'Whisper':
        model = Whisper(args)
        with open(args.save_dir + 'best_model.bin', 'rb') as file:
            model = pickle.load(file)
    
    elif args.model == 'DF':
        args.d = args.max_flow_length * 2

        INPUT_SHAPE = [2, args.max_flow_length]
        model = DFNet(INPUT_SHAPE, args.labels_num)
        load_or_initialize_parameters(model, args.save_dir + 'best_model.bin')
        model = model.to(args.device)
        
    elif args.model == 'YaTC':
        args.d = (args.header_bytes + args.payload_bytes) * args.input_packets

        model = models_YaTC.__dict__['TraFormer_YaTC'](
            num_classes=args.labels_num,
            drop_path_rate=args.drop_path,
        )
        load_or_initialize_parameters(model, args.save_dir + 'best_model.bin')
        model = model.to(args.device)
        
    elif args.model == 'TrafficFormer':
        args.d = args.seq_length

        args = load_hyperparam(args, args.bert_config_path)
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        model = Classifier(args)
        load_or_initialize_parameters(model, args.save_dir + 'best_model.bin')
        model = model.to(args.device)
    
    return args, model


def smoothing_joint(flow, sel_pos, packets, smoothing_params, pcap_level=False):
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
    new_packet_num = len(sel_pos)
    
    # time delay
    timestamp = flow['timestamp']
    iat = [0] + [timestamp[i] - timestamp[i-1] for i in range(1, flow['packet_num'])]
    new_iat = [iat[pos] for pos in sel_pos]
    beta_time_ms = smoothing_params['beta_time_ms']
    if beta_time_ms is not None:
        assert beta_time_ms > 0
        new_iat = [t + abs(np.random.normal(0, beta_time_ms)) * 0.001 for t in new_iat]
    new_timestamp = [0]
    for t in new_iat[1:]:
        new_timestamp.append(new_timestamp[-1] + float(t))
    
    # length padding
    pad = [0] * new_packet_num
    beta_length = smoothing_params['beta_length']
    if beta_length is not None:
        assert beta_length > 0
        pad = [int(abs(np.random.normal(0, beta_length))) for _ in range(new_packet_num)]
    direction_length = flow['direction_length']
    new_direction_length = [direction_length[pos] for pos in sel_pos]
    new_direction_length = [(length + np.sign(length) * pad[i]) 
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


@torch.no_grad()
def test(args, model, test_flows):
    if args.model not in ['kFP', 'Whisper']:
        model.eval()

    smoothing_times = []
    inference_times = []
    for i in tqdm(range(len(test_flows)), desc='Testing flows'):
        flow = test_flows[i]
        packet_num = flow['packet_num']
        d_sel = packet_num

        time_0 = time.time()
    
        if args.model in ['DF', 'Whisper']:
            n_jobs = 8
        elif args.model in ['Kitsune', 'kFP']:
            n_jobs = 20
        else:
            n_jobs = 32
        # Smoothing & Data preprocessing
        if args.smoothed == 'CertTA':
            if args.pr_sel is not None:
                d_sel = int(np.ceil(args.pr_sel * packet_num))
            
            samples_sel_pos = []
            samples_packets = []
            for _ in range(args.samples_num):
                sel_pos = random.sample(list(range(packet_num)), d_sel)
                if args.pacp_level is True:
                    input_packets = 5
                    sel_pos = sel_pos[:input_packets]
                samples_sel_pos.append(sel_pos)
                
                if args.pcap_level is True:
                    packets = [flow['packets'][i]['IP'] for i in sel_pos]
                    samples_packets.append(packets)
                else:
                    samples_packets.append(None)
            if args.pcap_level is True:
                del flow['packets']
            
            def generate_smoothing_instance(samples_sel_pos_packets):
                sel_pos, packets = samples_sel_pos_packets
                sample = smoothing_joint(flow, sel_pos, packets, args.smoothing_params, args.pcap_level)
                return flow_preprocessing(sample, args)
            
            with Pool(processes=n_jobs) as pool:
                instances = pool.map(generate_smoothing_instance, [(samples_sel_pos[i], samples_packets[i]) for _ in range(args.samples_num)])

        elif args.smoothed == 'VRS':
            instance_flow = flow_preprocessing(flow, args)
            instance_flow = dimension_alignment(args, instance_flow)
            
            def generate_smoothing_instance_vrs(_):
                return smoothing_vrs(instance_flow, args.sigma_vrs, args.model)
            
            with Pool(processes=n_jobs) as pool:
                instances = pool.map(generate_smoothing_instance_vrs, range(args.samples_num))
                
        elif args.smoothed == 'BARS':
            instance_flow = flow_preprocessing(flow, args)
            instance_flow = dimension_alignment(args, instance_flow)
            cls = flow['label']
            instances = generate_smoothing_samples_bars(instance_flow, args.noise_generator_cls[cls], args.t_cls[cls], args.model, args.samples_num)
            
        else:
            instances = [flow_preprocessing(flow, args)]
        
        # Batching
        input_output_batch = []
        for _ in range(len(instances[0])):
            input_output_batch.append([])
        for instance in instances:
            for k in range(len(instance)):
                input_output_batch[k].append(instance[k])

        time_1 = time.time()

        # Traditional ML models 
        if args.model in ['kFP', 'Whisper']:
            pred = model.test(input_output_batch[0])
        # DL/Transformer-based models
        elif args.model == 'Kitsune':
            src = torch.tensor(input_output_batch[0]).to(args.device)
            logits, _ = model(src)
            _, pred = logits.topk(1, 1, True, True)
        elif args.model == 'DF':
            src = torch.tensor(input_output_batch[0]).to(args.device)
            logits = model(src)
            _, pred = logits.topk(1, 1, True, True)
        elif args.model == 'YaTC':
            with torch.cuda.amp.autocast():
                src = torch.stack(input_output_batch[0]).to(args.device)
                logits = model(src)
            _, pred = logits.topk(1, 1, True, True)
        elif args.model == 'TrafficFormer':
            input_output_batch = [torch.LongTensor(io).to(args.device) for io in input_output_batch]
            _, logits = model(input_output_batch[0], None, input_output_batch[2], None)
            _, pred = logits.topk(1, 1, True, True)  
            
        class_count = np.zeros([args.labels_num])
        for c in pred:
            class_count[int(c)] += 1
        c_A = np.argmax(class_count)
        p_A = class_count[c_A] / sum(class_count)
        
        time_end = time.time()
        
        smoothing_times.append(time_1 - time_0)
        inference_times.append(time_end - time_1)
    
    avg_smoothing_time = np.mean(smoothing_times)
    avg_inference_time = np.mean(inference_times)
    with open(args.result_dir + 'delay.txt', 'w') as fp:
        fp.writelines("| Average Smoothing time {:6.3f} | Average Inference time {:6.3f} | \n".format(avg_smoothing_time, avg_inference_time))

                
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model", default="DF", choices=['kFP', 'Kitsune', 'Whisper', 'DF', 'TrafficFormer', 'YaTC'])
    parser.add_argument("--augment", type=str, default='CertTA', choices=['CertTA', 'VRS', 'BARS'],
                        help='train with the smoothing samples (perturbed flows)')
    parser.add_argument("--smoothed", type=str, default='CertTA', choices=['CertTA', 'VRS', 'BARS'],
                        help='test with randomized smoothing')
    smoothing_opts(parser)
    attack_opts(parser)
    training_opts(parser)
    args = parser.parse_args()
    print('Meassuring the delay of the {}-certified {} model.'.format(args.smoothed, args.model))
    

    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './evaluation/config/{}_{}_config.json'.format(args.model, args.dataset))
    
    if args.smoothed == 'CertTA': # Parameters for smoothing samples generation
        args.smoothing_params = {
            'beta_length': args.beta_length,
            'beta_time_ms': args.beta_time_ms,
            'pr_sel': args.pr_sel,
        }

    args.save_dir = './model/{}/save/{}/{}/'.format(args.model, args.dataset, model_name_generator(args))
    print('save dir: {}'.format(args.save_dir))
    args.result_dir = args.save_dir + ('base/' if args.smoothed is None else '{}/'.format(args.smoothed))
    print('smoothed:', args.smoothed)
    if args.attack is None:
        args.result_dir += 'clean/'
        print('clean test set')
    else:
        attack_name = attack_name_generator(args)
        args.attack_set_path = './attack/{}/{}/{}/attack.json'.format(args.attack, args.dataset, attack_name)
        args.result_dir += '{}/{}/'.format(args.attack, attack_name)
        print('attack name:', attack_name)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    print('--------------------------------------')

    print('Loading the dataset.')
    args.dataset_dir = './dataset/{}/json/'.format(args.dataset)
    with open(args.dataset_dir + 'statistics.json') as fp:
        statistics_json = json.load(fp)
    args.labels_num = statistics_json['label_num']
    args.pcap_level = args.model in ['YaTC', 'TrafficFormer']
    args.max_flow_length = {'CICDOH20': 100, 'TIISSRC23': 100}[args.dataset]
    test_flows = load_data(args.dataset_dir + 'test.json', args.pcap_level, args.max_flow_length)
    args.test_flows_num = len(test_flows)
    if args.attack is not None:
        test_flows = generate_attack_flows(args, test_flows)

    print('Building the traffic analysis model.')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(args.device))
    args, model = initialize_model(args)
    if args.model in ['kFP', 'Kitsune']:
        normalizer = Normalizer()
        normalizer.norm_max = np.load(args.save_dir + "norm_max.npy")
        normalizer.norm_min = np.load(args.save_dir + "norm_min.npy")
        args.normalizer = normalizer

    if args.augment == 'BARS':
        print('Loading BARS noise generators')
        args.bars_dir = './model/{}/save/{}/BARS/'.format(args.model, args.dataset)
        print('bars dir: {}'.format(args.bars_dir))
        args.noise_generator_cls = []
        args.t_cls = []
        for i in range(args.labels_num):
            args.feature_noise_distribution = 'gaussian'
            noise_generator = create_noise_generator(args)
            noise_generator.distribution_transformer = torch.load(os.path.join(args.bars_dir, "checkpoint-distribution-transformer-" + str(i))).to(args.device)
            args.noise_generator_cls.append(noise_generator)

            r = open(os.path.join(args.bars_dir, "t-" + str(i)), "r")
            t = float(r.readline())
            r.close()
            args.t_cls.append(t)

    test(args, model, test_flows)


if __name__ == '__main__':
    main()