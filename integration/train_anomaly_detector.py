import os
import sys
import time
import random
import argparse
from tqdm import tqdm
import pytorch_warmup

sys.path.append('.') # run in directory: CertTA_public/
from evaluation.utilities import *
from evaluation.opts import training_opts
from evaluation.data_loader import load_data, flow_preprocessing

sys.path.append('./model/KMeans')
from model.KMeans.model_KMeans_AD import KMeans_AD

sys.path.append('./model/Whisper')
from model.Whisper.model_Whisper_AD import Whisper_AD

sys.path.append('./model/Kitsune')
from model.Kitsune.KitNET_AD import KitNET_AD


def initialize_model(args):
    if args.model == 'KMeans':
        model = KMeans_AD(args)
        model.clear_train_data()
        return args, model, None, None
    
    elif args.model == 'Whisper':
        model = Whisper_AD(args)
        model.clear_train_data()
        return args, model, None, None
    
    elif args.model == 'Kitsune':
        model = KitNET_AD(args.feature_num, args.labels_num, args.maxAE, feature_map=[[i for i in range(args.feature_num)]], device=args.device)
        model = model.to(args.device)
        parameters = []
        for ae in model.ensembleLayer:
            parameters += list(ae.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        args.train_steps = int(args.train_flows_num * args.epochs_num / args.batch_size) + 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps)
        args.warmup_steps = int(args.train_flows_num * args.epochs_num * args.warmup / args.batch_size)
        args.warmup_scheduler =  pytorch_warmup.LinearWarmup(optimizer, args.warmup_steps)
        return args, model, optimizer, scheduler


def create_normalizer(args, train_flows):
    normalizer = Normalizer()
    feats = []
    feats = []
    for flow in tqdm(train_flows, desc='Flow Preprocessing (for normalizer)', disable=False):
        src_feat, _ = flow_preprocessing(flow, args)
        feats.append(np.array(src_feat))
    normalizer.update(np.array(feats)) 
    np.save(args.save_dir + "norm_max.npy", normalizer.norm_max)
    np.save(args.save_dir + "norm_min.npy", normalizer.norm_min)
    return normalizer


def train_one_epoch(args, model, optimizer, scheduler, train_flows, epoch_idx):
    if args.model in ['Kitsune']:
        model.train()
        optimizer.zero_grad()
    train_loss = 0
    time_start = time.time()
    
    train_distances = []
    
    # Data preprocessing
    random.shuffle(train_flows)
    instances = [] # instances: [(x, y, ...), ...]
    for flow in tqdm(train_flows, desc='Flow Preprocessing', disable=False):
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
            model.add_train_data(input_output_batch[0])
        # DL/Transformer-based models
        elif args.model == 'Kitsune':
            src = torch.tensor(input_output_batch[0]).to(args.device)
            loss_rec = model(src)
            loss = loss_rec.mean()
            loss.backward()

            optimizer.step()
            with args.warmup_scheduler.dampening():
                scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.detach().item()
            train_distances_batch = loss_rec.cpu().detach().numpy().tolist()
            train_distances.extend(train_distances_batch)
    
    if args.model in ['KMeans', 'Whisper']:
        train_distances = model.train()
        train_loss = np.mean(train_distances)
        model.clear_train_data()
    elif args.model == 'Kitsune':
        model.train_distances = train_distances
    
    print('Train_loss', train_loss, 'Cost Time', time.time() - time_start)
    return train_loss

@torch.no_grad()
def evaluate(args, model, eval_flows):
    if args.model in ['Kitsune']:
        model.eval()
    eval_loss = 0
    time_start = time.time()
    
    # Data preprocessing
    instances = [] # instances: [(x, y, ...), ...]
    for flow in tqdm(eval_flows, desc='Flow Preprocessing', disable=False):
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
            eval_distances = model.test(input_output_batch[0])
            eval_loss = np.mean(eval_distances)
        # DL/Transformer-based models
        elif args.model == 'Kitsune':
            src = torch.tensor(input_output_batch[0]).to(args.device)
            loss_rec = model(src)
            loss = loss_rec.mean()
            
            eval_loss += loss.detach().item()
    
    print('Eval_loss', eval_loss, 'Cost Time', time.time() - time_start)
    return eval_loss
    
        
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model", default="Kitsune", choices=['KMeans', 'Whisper', 'Kitsune'])
    parser.add_argument("--replace_best_model", type=bool, default=True, help='replace the best model')
    parser.add_argument("--truncate", type=float, default=None, choices=[None, 0.25, 0.5, 0.75])
    training_opts(parser)
    args = parser.parse_args()
    print('Training the {}_AD model.'.format(args.model))
    
    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './AnomalyDetection/config/{}_AD_{}_config.json'.format(args.model, args.dataset))
    
    args.save_dir = './model/{}/save/{}/{}_AD{}/'.format(args.model, args.dataset, args.model, '_truncate_{}'.format(args.truncate) if args.truncate is not None else '')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print('save dir: {}'.format(args.save_dir))
    print('--------------------------------------')
    
    print('Loading the dataset.')
    args.dataset_dir = './dataset/{}/json/'.format(args.dataset)
    with open(args.dataset_dir + 'statistics.json') as fp:
        statistics_json = json.load(fp)
    args.labels_num = statistics_json['label_num']
    args.pcap_level = args.model in ['YaTC', 'TrafficFormer']
    args.max_flow_length = {'CICDOH20': 100, 'TIISSRC23': 100}[args.dataset]
    
    train_flows = load_data(args.dataset_dir + 'train.json', args.pcap_level, args.max_flow_length, args.truncate)
    valid_flows = load_data(args.dataset_dir + 'valid.json', args.pcap_level, args.max_flow_length, args.truncate)
    args.train_flows_num = len(train_flows)
    
    print('Building the anomaly detection model.')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, model, optimizer, scheduler = initialize_model(args)
    print(torch.cuda.get_device_name(args.device))
    if args.model in ['KMeans', 'Kitsune']:
        args.normalizer = create_normalizer(args, train_flows)

    print('Start training.')
    best_round = 0
    best_valid_loss = np.inf
    for epoch in tqdm(range(1, args.epochs_num + 1), desc='Training Epoch'):
        train_loss = train_one_epoch(args,
            model, optimizer, scheduler,
            train_flows, epoch
        )

        valid_loss = evaluate(args, model, valid_flows)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_round = epoch
            save_model(args.model, model, args.save_dir + 'best_model_{}.bin'.format(os.getpid()))
            if args.replace_best_model:
                save_model(args.model, model, args.save_dir + 'best_model.bin')
        elif epoch - best_round >= args.early_stop:
            print("Early Stopping!")
            logging(args.save_dir + 'training_log_{}.txt'.format(os.getpid()), 'Early Stopping!\n')
            break

        logging(args.save_dir + 'training_log_{}.txt'.format(os.getpid()), 
                'Epoch {}, best valid loss {}, train loss {}, valid loss {}.\n'.format(
                    epoch, best_valid_loss, train_loss, valid_loss)
        )


if __name__ == '__main__':
    main()