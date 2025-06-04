import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

import sys
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append('.') # run in directory: CertTA_public/
from evaluation.utilities import *
from evaluation.opts import smoothing_opts, attack_opts, training_opts
from evaluation.data_loader import load_data, generate_attack_flows, flow_preprocessing, dimension_alignment
from certification.smoothing import generate_smoothing_samples, generate_smoothing_samples_vrs, generate_smoothing_samples_bars
from BARS.train_noise_generator import create_noise_generator

sys.path.append('./model/KMeans')
from model.KMeans.model_KMeans_AD import KMeans_AD

sys.path.append('./model/kFP')
from model.kFP.model_kFP import kFP

sys.path.append('./model/Kitsune')
from model.Kitsune.KitNET import KitNET
from model.Kitsune.KitNET_AD import KitNET_AD

sys.path.append('./model/Whisper')
from model.Whisper.model_Whisper import Whisper
from model.Whisper.model_Whisper_AD import Whisper_AD

sys.path.append('./model/DF')
from model.DF.DFNet import DFNet

sys.path.append('./model/YaTC')
import model.YaTC.models_YaTC as models_YaTC

sys.path.append('./model/TrafficFormer')
from model.TrafficFormer.fine_tuning.run_classifier import Classifier
from model.TrafficFormer.uer.utils import *


def initialize_model_AD(args):
    if args.model_AD == 'KMeans':
        model_AD = KMeans_AD(args)
        with open(args.save_dir_AD + 'best_model.bin', 'rb') as file:
            model_AD = pickle.load(file)
    
    elif args.model_AD == 'Whisper':
        model_AD = Whisper_AD(args)
        with open(args.save_dir_AD + 'best_model.bin', 'rb') as file:
            model_AD = pickle.load(file)
    
    elif args.model_AD == 'Kitsune':
        model_AD = KitNET_AD(args.feature_num, args.labels_num, args.maxAE, feature_map=[[i for i in range(args.feature_num)]], device=args.device)
        with open(args.save_dir_AD + 'best_model.bin', 'rb') as file:
            model_AD = pickle.load(file)
        model_AD = model_AD.to(args.device)
    
    return args, model_AD


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


@torch.no_grad()
def anomaly_detection(args, model_AD, test_flows, distance_threshold):
    normal_flows = []
    anomalies = []
    
    if args.model_AD in ['Kitsune']:
        model_AD.eval()
    
    # Data preprocessing
    instances = [] # instances: [(x, y, ...), ...]
    for flow in tqdm(test_flows, desc='Flow Preprocessing', disable=False):
        instance_flow = flow_preprocessing(flow, args)
        if instance_flow == -1:
            raise Exception('Bad Flow:' + flow['source_file'])
        instances.append(instance_flow)
    
    for batch_idx in tqdm(range(math.ceil(len(instances) / args.batch_size)), desc='Batch', disable=False):
        # Batching
        input_output_batch = [] # input_output_batch: [(x_1, x_2, ...), (y_1, y_2, ...), ...]
        for _ in range(len(instances[0])):
            input_output_batch.append([])
        for instance_idx in range(batch_idx * args.batch_size, min(len(instances), (batch_idx + 1) * args.batch_size)):
            instance = instances[instance_idx]
            for k in range(len(instance)):
                input_output_batch[k].append(instance[k])

        # Traditional ML models
        if args.model in ['KMeans', 'Whisper']:
            test_distances = model_AD.test(input_output_batch[0])
        # DL/Transformer-based models
        elif args.model == 'Kitsune':
            src = torch.tensor(input_output_batch[0]).to(args.device)
            test_distances = model_AD(src)
        
        for i, instance_idx in enumerate(range(batch_idx * args.batch_size, min(len(instances), (batch_idx + 1) * args.batch_size))):
            test_flows[instance_idx]['AD_distance'] = test_distances[i]
            if test_distances[i] >= distance_threshold:
                anomalies.append(test_flows[instance_idx])
            else:
                normal_flows.append(test_flows[instance_idx])
    
    return normal_flows, anomalies


@torch.no_grad()
def one_of_k_test(args, model, test_flows, ith):
    if args.model not in ['kFP', 'Whisper']:
        model.eval()
    
    res = []
    for i in tqdm(range(len(test_flows)), desc='Testing flows'):
        flow = test_flows[i]
            
        # Smoothing & Data preprocessing
        if args.smoothed == 'CertTA':
            smoothing_samples = generate_smoothing_samples(flow, args.smoothing_params, args.samples_num, args.pcap_level)
            instances = [flow_preprocessing(sample, args) for sample in smoothing_samples]
        elif args.smoothed == 'VRS':
            instance_flow = flow_preprocessing(flow, args)
            instance_flow = dimension_alignment(args, instance_flow)
            instances = generate_smoothing_samples_vrs(instance_flow, args.sigma_vrs, args.model, args.samples_num)
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
        
        if args.smoothed in ['VRS', 'BARS']:
            if args.model == 'YaTC':
                input_vector = instance_flow[2]
            elif args.model == 'TrafficFormer' and args.smoothed == 'VRS':
                input_vector = instance_flow[3]
            else:
                input_vector = instance_flow[0]
            if not isinstance(input_vector, list):
                input_vector = input_vector.tolist()
        else:
            input_vector = None
        
        res.append({
            'label': flow['label'],
            'c_A': int(c_A),
            'p_A': p_A,
            'packet_num': flow['packet_num'],
            'input_vector': str(input_vector),
            'source_file': flow['source_file']
        })
        
    with open(args.result_dir + '{}th.json'.format(ith), 'w') as fp:
        json.dump(res, fp, indent=1)

                
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model_AD", default="Kitsune", choices=['KMeans', 'Whisper', 'Kitsune'])
    parser.add_argument("--FPR_threshold", type=float, default=0.01)
    parser.add_argument("--model", default="YaTC", choices=['kFP', 'Kitsune', 'Whisper', 'DF', 'TrafficFormer', 'YaTC'])
    parser.add_argument("--augment", type=str, default='CertTA', choices=['CertTA', 'VRS', 'BARS'],
                        help='train with the smoothing samples (perturbed flows)')
    parser.add_argument("--smoothed", type=str, default='CertTA', choices=['CertTA', 'VRS', 'BARS'],
                        help='test with randomized smoothing')
    parser.add_argument("--truncate", type=float, default=None, choices=[None, 0.25, 0.5, 0.75])
    smoothing_opts(parser)
    attack_opts(parser)
    training_opts(parser)
    args = parser.parse_args()
    print('Testing the integration of the {}_AD model and the {}-certified {} model.'.format(args.model_AD, args.smoothed, args.model))

    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './evaluation/config/{}_{}_config.json'.format(args.model, args.dataset))
    
    if args.smoothed == 'CertTA': # Parameters for smoothing samples generation
        args.smoothing_params = {
            'beta_length': args.beta_length,
            'beta_time_ms': args.beta_time_ms,
            'pr_sel': args.pr_sel,
        }
    elif args.smoothed == 'BARS' or args.augment == 'BARS':
        if model in ['kFP', 'Whisper']:
            raise Exception('BARS is not applicable to the {} model'.format(args.model))

    args.save_dir_AD = './model/{}/save/{}/{}_AD{}/'.format(args.model_AD, args.dataset, args.model_AD, '_truncate_{}'.format(args.truncate) if args.truncate is not None else '')
    args.save_dir = './model/{}/save/{}/{}{}/'.format(args.model, args.dataset, model_name_generator(args), '_truncate_{}'.format(args.truncate) if args.truncate is not None else '')
    print('save dir: {}'.format(args.save_dir))
    args.result_dir = args.save_dir + ('base_with_AD/' if args.smoothed is None else '{}_with_AD/'.format(args.smoothed))
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
    test_flows = load_data(args.dataset_dir + 'test.json', args.pcap_level, args.max_flow_length, args.truncate)
    args.test_flows_num = len(test_flows)
    if args.attack is not None:
        test_flows = generate_attack_flows(args, test_flows)

    print('Building the anomaly detection model.')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, model_AD = initialize_model_AD(args)
    print(torch.cuda.get_device_name(args.device))
    if args.model in ['KMeans', 'Kitsune']:
        normalizer_AD = Normalizer()
        normalizer_AD.norm_max = np.load(args.save_dir + "norm_max.npy")
        normalizer_AD.norm_min = np.load(args.save_dir + "norm_min.npy")
        args.normalizer = normalizer_AD
    
    # Set the distance threshold according to FPR threshold
    train_distances = model_AD.train_distances
    train_distances = sorted(train_distances, reverse=True)
    distance_threshold = train_distances[int(np.floor(len(train_distances) * args.FPR_threshold))]

    print('Performing anomaly detection: threshold {}.'.format(distance_threshold))
    normal_flows, anomalies = anomaly_detection(args, model_AD, test_flows, distance_threshold)
    anomalies_res = [{
        'label': flow['label'],
        'c_A': -1, # c_A = -1 for anomalies
        'AD_distance': flow['AD_distance'],
        'packet_num': flow['packet_num'],
        'source_file': flow['source_file']
    } for flow in anomalies]
    with open(args.result_dir + 'anomalies.json', 'w') as fp:
        json.dump(anomalies_res, fp, indent=1)
    
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

    print('Performing traffic analysis.')
    # one_of_k_test(args, model, test_flows, 0)
    k = 24 if args.model in ['Whisper'] else 8
    print('Building process pool...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    pool = Pool(processes=k)
    args.normal_flows_num = len(normal_flows)
    for i in range(0, k):
        test_flows_process = normal_flows[int(i * args.normal_flows_num / k): int((i+1) * args.normal_flows_num / k)]
        pool.apply_async(one_of_k_test, (args, model, test_flows_process, i), error_callback=print_error)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()