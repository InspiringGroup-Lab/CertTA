import os
import sys
import argparse
from tqdm import tqdm

sys.path.append('.') # run in directory: CertTA_public/
from evaluation.utilities import *
from evaluation.opts import training_opts, attack_opts
from evaluation.data_loader import load_data, flow_preprocessing, generate_attack_flows

sys.path.append('./model/KMeans')
from model.KMeans.model_KMeans_AD import KMeans_AD

sys.path.append('./model/Whisper')
from model.Whisper.model_Whisper_AD import Whisper_AD

sys.path.append('./model/Kitsune')
from model.Kitsune.KitNET_AD import KitNET_AD


def initialize_model(args):
    if args.model == 'KMeans':
        model = KMeans_AD(args)
        with open(args.save_dir + 'best_model.bin', 'rb') as file:
            model = pickle.load(file)
            
    elif args.model == 'Whisper':
        model = Whisper_AD(args)
        with open(args.save_dir + 'best_model.bin', 'rb') as file:
            model = pickle.load(file)
            
    elif args.model == 'Kitsune':
        model = KitNET_AD(args.feature_num, args.labels_num, args.maxAE, feature_map=[[i for i in range(args.feature_num)]], device=args.device)
        with open(args.save_dir + 'best_model.bin', 'rb') as file:
            model = pickle.load(file)
        model = model.to(args.device)
    
    return args, model


@torch.no_grad()
def anomaly_detection(args, model, test_flows, distance_threshold):
    normal_flows = []
    anomalies = []
    
    if args.model in ['Kitsune']:
        model.eval()
    
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
            test_distances = model.test(input_output_batch[0])
        # DL/Transformer-based models
        elif args.model == 'Kitsune':
            src = torch.tensor(input_output_batch[0]).to(args.device)
            test_distances = model(src)
        
        for i, instance_idx in enumerate(range(batch_idx * args.batch_size, min(len(instances), (batch_idx + 1) * args.batch_size))):
            test_flows[instance_idx]['AD_distance'] = test_distances[i]
            if test_distances[i] >= distance_threshold:
                anomalies.append(test_flows[instance_idx])
            else:
                normal_flows.append(test_flows[instance_idx])
    
    return normal_flows, anomalies


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model", default="Kitsune", choices=['KMeans', 'Whisper', 'Kitsune'])
    parser.add_argument("--FPR_threshold", type=float, default=0.01)
    parser.add_argument("--truncate", type=float, default=None, choices=[None, 0.25, 0.5, 0.75])
    attack_opts(parser)
    training_opts(parser)
    args = parser.parse_args()
    print('Testing the {}_AD model.'.format(args.model))
    
    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './AnomalyDetection/config/{}_AD_{}_config.json'.format(args.model, args.dataset))
    
    args.save_dir = './model/{}/save/{}/{}_AD{}/'.format(args.model, args.dataset, args.model, args.model_AD, '_truncate_{}'.format(args.truncate) if args.truncate is not None else '')
    print('save dir: {}'.format(args.save_dir))
    if args.attack is None:
        args.result_dir = args.save_dir + 'clean/'
        print('clean test set')
    else:
        attack_name = attack_name_generator(args)
        args.attack_set_path = './attack/{}/{}/{}/attack.json'.format(args.attack, args.dataset, attack_name)
        args.result_dir = args.save_dir + '{}/{}/'.format(args.attack, attack_name)
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
    args, model = initialize_model(args)
    print(torch.cuda.get_device_name(args.device))
    if args.model in ['KMeans', 'Kitsune']:
        normalizer = Normalizer()
        normalizer.norm_max = np.load(args.save_dir + "norm_max.npy")
        normalizer.norm_min = np.load(args.save_dir + "norm_min.npy")
        args.normalizer = normalizer

    # Set the distance threshold according to FPR threshold
    train_distances = model.train_distances
    train_distances = sorted(train_distances, reverse=True)
    distance_threshold = train_distances[int(np.floor(len(train_distances) * args.FPR_threshold))]

    normal_flows, anomalies = anomaly_detection(args, model, test_flows, distance_threshold)
    with open(args.result_dir + 'anomaly_detection_acc.txt', 'w') as fp:
        fp.writelines("| FPR {:6.3f}, PR {:6.3f}\n".format(
            np.floor(len(train_distances) * args.FPR_threshold) / len(train_distances), 
            len(anomalies) / len(test_flows)))


if __name__ == '__main__':
    main()