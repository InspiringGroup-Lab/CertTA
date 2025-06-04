import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

import sys
import time
import json
import math
import torch
import random
import torch.backends
import argparse
from tqdm import tqdm
import pytorch_warmup

sys.path.append('.') # run in directory: CertTA_public/
from evaluation.utilities import *
from evaluation.opts import smoothing_opts, training_opts
from evaluation.data_loader import load_data, flow_preprocessing, dimension_alignment
from certification.smoothing import smoothing_joint, smoothing_vrs, smoothing_bars
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
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
from model.YaTC.model_loader import load_checkpoint
from timm.loss import LabelSmoothingCrossEntropy

sys.path.append('./model/TrafficFormer')
from model.TrafficFormer.fine_tuning.run_classifier import Classifier, build_optimizer
from model.TrafficFormer.uer.utils import *


def initialize_model(args):
    if args.model == 'kFP':
        model = kFP(args)
        model.clear_train_data()
        return args, model, None, None
    
    elif args.model == 'Whisper':
        model = Whisper(args)
        model.clear_train_data()
        return args, model, None, None
    
    elif args.model == 'DF':
        args.d = args.max_flow_length * 2

        INPUT_SHAPE = [2, args.max_flow_length]
        model = DFNet(INPUT_SHAPE, args.labels_num)
        model = model.to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        args.train_steps = int(args.train_flows_num * args.epochs_num / args.batch_size) + 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps)
        args.warmup_steps = int(args.train_flows_num * args.epochs_num * args.warmup / args.batch_size)
        args.warmup_scheduler =  pytorch_warmup.LinearWarmup(optimizer, args.warmup_steps)
        return args, model, optimizer, scheduler
    
    elif args.model == 'YaTC':
        args.d = (args.header_bytes + args.payload_bytes) * args.input_packets
        
        model = models_YaTC.__dict__['TraFormer_YaTC'](
            num_classes=args.labels_num,
            drop_path_rate=args.drop_path,
        )
        model = load_checkpoint(model, args.pretrained_model_path)
        model = model.to(args.device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model = %s" % str(model))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        args.learning_rate = args.base_learning_rate * args.batch_size / 256
        print("base lr: %.2e" % (args.base_learning_rate))
        print("actual lr: %.2e" % args.learning_rate)
        param_groups = param_groups_lrd(model, args.weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=args.layer_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)
        return args, model, optimizer, None
    
    elif args.model == 'TrafficFormer':
        args.d = args.seq_length

        args = load_hyperparam(args, args.bert_config_path)
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        model = Classifier(args)
        load_or_initialize_parameters(model, args.pretrained_model_path, {'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})
        model = model.to(args.device)
        args.train_steps = int(args.train_flows_num * args.epochs_num / args.batch_size) + 1
        optimizer, scheduler = build_optimizer(args, model)
        return args, model, optimizer, scheduler
    
    
def initialize_model_Kitsune(args, train_flows):
    args.d = args.feature_num
    model = KitNET(args.feature_num, args.labels_num, args.maxAE, feature_map=[[i for i in range(args.feature_num)]], device=args.device)

    model = model.to(args.device)
    model_params = list(model.outputLayer.parameters())
    for ae in model.ensembleLayer:
        model_params += list(ae.parameters())
    optimizer = torch.optim.AdamW(model_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    args.train_steps = int(args.train_flows_num * args.epochs_num / args.batch_size) + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps)
    args.warmup_steps = int(args.train_flows_num * args.epochs_num * args.warmup / args.batch_size)
    args.warmup_scheduler =  pytorch_warmup.LinearWarmup(optimizer, args.warmup_steps)
    return args, model, optimizer, scheduler


def create_normalizer(args, train_flows):
    normalizer = Normalizer()
    feats = []
    if args.augment != 'CertTA':
        feats = []
        for flow in tqdm(train_flows, desc='Flow Preprocessing (for normalizer)', disable=False):
            src_feat, _ = flow_preprocessing(flow, args)
            feats.append(np.array(src_feat))
        normalizer.update(np.array(feats)) 
    else:
        for _ in range(5):
            feats = []
            for flow in tqdm(train_flows, desc='Flow Preprocessing (for normalizer)', disable=False):
                flow = smoothing_joint(flow, args.smoothing_params, args.pcap_level)
                src_feat, _ = flow_preprocessing(flow, args)
                feats.append(np.array(src_feat))
            normalizer.update(np.array(feats))
        
    np.save(args.save_dir + "norm_max.npy", normalizer.norm_max)
    np.save(args.save_dir + "norm_min.npy", normalizer.norm_min)
    return normalizer


def train_one_epoch(args, model, optimizer, scheduler, train_flows, epoch_idx):
    if args.model not in ['kFP', 'Whisper']:
        model.train()
        optimizer.zero_grad()
    train_loss = 0
    time_start = time.time()

    # Data preprocessing
    random.shuffle(train_flows)
    instances = [] # instances: [(x, y, ...), ...]
    for flow in tqdm(train_flows, desc='Flow Preprocessing', disable=False):
        if args.augment == 'CertTA':
            flow = smoothing_joint(flow, args.smoothing_params, args.pcap_level)
        instance_flow = flow_preprocessing(flow, args)
        if instance_flow == -1:
            raise Exception('Bad Flow:' + flow['source_file'])

        if args.augment == 'VRS':
            instance_flow = dimension_alignment(args, instance_flow)
            instance_flow = smoothing_vrs(instance_flow, args.sigma_vrs, args)
        if args.augment == 'BARS':
            instance_flow = dimension_alignment(args, instance_flow)
            instance_flow = smoothing_bars(instance_flow, args.noise_generator_cls[instance_flow[1]], args.t_cls[instance_flow[1]], args.model)
            
        instances.append(instance_flow)

    y_true = []
    y_pred = []
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
        if args.model in ['kFP', 'Whisper']:
            model.add_train_data(input_output_batch[0], input_output_batch[1])
        # DL/Transformer-based models
        elif args.model == 'Kitsune':
            y_true.extend(input_output_batch[1])

            src = torch.tensor(input_output_batch[0]).to(args.device)
            tgt = torch.tensor(input_output_batch[1]).to(args.device)
            logits, loss_rec = model(src)
            loss_clf = torch.nn.CrossEntropyLoss()(logits, tgt)
            loss = loss_clf + loss_rec * args.beta
            loss.backward()

            optimizer.step()
            with args.warmup_scheduler.dampening():
                scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.detach().item()
            pred = logits.argmax(dim=1, keepdim=True)
            y_pred.extend(pred.t()[0].cpu())
        elif args.model == 'DF':
            y_true.extend(input_output_batch[1])
            
            src = torch.tensor(input_output_batch[0]).to(args.device)
            tgt = torch.tensor(input_output_batch[1]).to(args.device)
            logits = model(src)
            loss = torch.nn.CrossEntropyLoss()(logits, tgt)
            loss.backward()
            
            optimizer.step()
            with args.warmup_scheduler.dampening():
                scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.detach().item()
            _, pred = logits.topk(1, 1, True, True)
            y_pred.extend(pred.t()[0].cpu())
        elif args.model == 'YaTC':
            y_true.extend(input_output_batch[1])

            adjust_learning_rate(optimizer, 
                                 epoch_idx + batch_idx * args.batch_size / len(instances), 
                                 args.epochs_num, args.warmup_epochs,
                                 args.learning_rate, args.min_learning_rate)

            if args.label_smoothing > 0.:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

            with torch.cuda.amp.autocast():
                src = torch.stack(input_output_batch[0]).to(args.device)
                tgt = torch.tensor(input_output_batch[1], device=args.device)
                logits = model(src)
                loss = criterion(logits, tgt)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.detach().item()
            _, pred = logits.topk(1, 1, True, True)
            y_pred.extend(pred.t()[0].cpu())
        elif args.model == 'TrafficFormer':
            y_true.extend(input_output_batch[1])

            input_output_batch = [torch.LongTensor(io).to(args.device) for io in input_output_batch]
            loss, logits = model(input_output_batch[0], input_output_batch[1], input_output_batch[2], None)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.detach().item()
            _, pred = logits.topk(1, 1, True, True)
            y_pred.extend(pred.t()[0].cpu())

    if args.model in ['kFP', 'Whisper']:
        if args.augment is not None and args.augment_times > 0:
            args.augment_times -= 1
            return train_one_epoch(args, model, optimizer, scheduler, train_flows, epoch_idx)
        y_true, y_pred = model.train()
        model.clear_train_data()

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')

    print('Train_loss', train_loss, 'Precision', precision, 'Recall', recall, 'F1', f1score, 'Cost Time', time.time() - time_start)
    print(confusion_matrix(y_true, y_pred))

    return f1score, train_loss


@torch.no_grad()
def evaluate(args, model, eval_flows):
    if args.model not in ['kFP', 'Whisper']:
        model.eval()
    eval_loss = 0
    time_start = time.time()

    # Data preprocessing
    instances = [] # instances: [(x, y, ...), ...]
    for flow in tqdm(eval_flows, desc='Flow Preprocessing', disable=True):
        if args.augment == 'CertTA':
            flow = smoothing_joint(flow, args.smoothing_params, args.pcap_level)
        instance_flow = flow_preprocessing(flow, args)
        if instance_flow == -1:
            raise Exception('Bad Flow:' + flow['source_file'])

        if args.augment == 'VRS':
            instance_flow = dimension_alignment(args, instance_flow)
            instance_flow = smoothing_vrs(instance_flow, args.sigma_vrs, args)
        if args.augment == 'BARS':
            instance_flow = dimension_alignment(args, instance_flow)
            instance_flow = smoothing_bars(instance_flow, args.noise_generator_cls[instance_flow[1]], args.t_cls[instance_flow[1]], args.model)   
            
        instances.append(instance_flow)

    y_true = []
    y_pred = []
    for batch_idx in tqdm(range(math.ceil(len(instances) / args.batch_size)), desc='Batch', disable=False):
        # Batching
        input_output_batch = []
        for _ in range(len(instances[0])):
            input_output_batch.append([])
        for instance_idx in range(batch_idx * args.batch_size, min(len(instances), (batch_idx + 1) * args.batch_size)):
            instance = instances[instance_idx]
            for k in range(len(instance)):
                input_output_batch[k].append(instance[k])

        # Traditional ML models 
        if args.model in ['kFP', 'Whisper']:
            y_true.extend(input_output_batch[1])

            pred = model.test(input_output_batch[0])
            y_pred.extend(pred.tolist())
        # DL/Transformer-based models
        elif args.model == 'Kitsune':
            y_true.extend(input_output_batch[1])

            src = torch.tensor(input_output_batch[0]).to(args.device)
            tgt = torch.tensor(input_output_batch[1]).to(args.device)
            logits, loss_rec = model(src)
            loss_clf = torch.nn.CrossEntropyLoss()(logits, tgt)
            loss = loss_clf + loss_rec * args.alpha

            eval_loss += loss.detach().item()
            pred = logits.argmax(dim=1, keepdim=True)
            y_pred.extend(pred.t()[0].cpu())
        elif args.model == 'DF':
            y_true.extend(input_output_batch[1])
            
            src = torch.tensor(input_output_batch[0]).to(args.device)
            tgt = torch.tensor(input_output_batch[1]).to(args.device)
            logits = model(src)
            loss = torch.nn.CrossEntropyLoss()(logits, tgt)

            eval_loss += loss.detach().item()
            _, pred = logits.topk(1, 1, True, True)
            y_pred.extend(pred.t()[0].cpu())
        elif args.model == 'YaTC':
            y_true.extend(input_output_batch[1])

            if args.label_smoothing > 0.:
                criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

            with torch.cuda.amp.autocast():
                src = torch.stack(input_output_batch[0]).to(args.device)
                tgt = torch.tensor(input_output_batch[1], device=args.device)
                logits = model(src)
                loss = criterion(logits, tgt)

            eval_loss += loss.detach().item()
            _, pred = logits.topk(1, 1, True, True)
            y_pred.extend(pred.t()[0].cpu())
        elif args.model == 'TrafficFormer':
            y_true.extend(input_output_batch[1])

            input_output_batch = [torch.LongTensor(io).to(args.device) for io in input_output_batch]
            loss, logits = model(input_output_batch[0], input_output_batch[1], input_output_batch[2], None)

            eval_loss += loss.detach().item()
            _, pred = logits.topk(1, 1, True, True)
            y_pred.extend(pred.t()[0].cpu())     
    
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')

    print('Eval_loss', eval_loss, 'Precision', precision, 'Recall', recall, 'F1', f1score, 'Cost Time', time.time() - time_start)
    print(confusion_matrix(y_true, y_pred))

    return f1score, eval_loss


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model", default="DF", choices=['kFP', 'Kitsune', 'Whisper', 'DF', 'YaTC', 'TrafficFormer'])
    parser.add_argument("--augment", type=str, default=None, choices=['CertTA', 'VRS', 'BARS'],
                        help='train with the smoothing samples (perturbed flows)')
    parser.add_argument("--replace_best_model", type=bool, default=True, help='replace the best model')
    parser.add_argument("--truncate", type=float, default=None, choices=[None, 0.25, 0.5, 0.75])
    smoothing_opts(parser)
    training_opts(parser)
    args = parser.parse_args()
    print('Training the base {} model.'.format(args.model))

    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './evaluation/config/{}_{}_config.json'.format(args.model, args.dataset))
    if args.augment == 'CertTA': # Parameters for smoothing samples generation
        args.smoothing_params = {
            'beta_length': args.beta_length,
            'beta_time_ms': args.beta_time_ms,
            'pr_sel': args.pr_sel,
        }
    args.save_dir = './model/{}/save/{}/{}{}/'.format(args.model, args.dataset, model_name_generator(args), '_truncate_{}'.format(args.truncate) if args.truncate is not None else '')
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

    print('Building the traffic analysis model.')
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == 'Kitsune':
        args, model, optimizer, scheduler = initialize_model_Kitsune(args, train_flows)
    else:
        args, model, optimizer, scheduler = initialize_model(args)
    print(torch.cuda.get_device_name(args.device))
    if args.model in ['kFP', 'Kitsune']:
        args.normalizer = create_normalizer(args, train_flows)
    
    if args.augment == 'BARS':
        if model in ['kFP', 'Whisper']:
            raise Exception('BARS is not applicable to the {} model'.format(args.model))
        print('Loading BARS noise generators.')
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
            
    print('Start training.')
    best_round = 0
    best_valid_f1_score = -1
    for epoch in tqdm(range(1, args.epochs_num + 1), desc='Training Epoch'):
        train_f1_score, train_loss = train_one_epoch(args,
            model, optimizer, scheduler,
            train_flows, epoch
        )

        valid_f1_score, valid_loss = evaluate(args, model, valid_flows)

        if valid_f1_score > best_valid_f1_score:
            best_valid_f1_score = valid_f1_score
            best_round = epoch
            save_model(args.model, model, args.save_dir + 'best_model_{}.bin'.format(os.getpid()))
            if args.replace_best_model:
                save_model(args.model, model, args.save_dir + 'best_model.bin')
        elif epoch - best_round >= args.early_stop:
            print("Early Stopping!")
            logging(args.save_dir + 'training_log_{}.txt'.format(os.getpid()), 'Early Stopping!\n')
            break

        logging(args.save_dir + 'training_log_{}.txt'.format(os.getpid()), 
                'Epoch {}, train f1 {}, valid f1 {}, best valid f1 {}, train loss {}, valid loss {}.\n'.format(
                    epoch, train_f1_score, valid_f1_score, best_valid_f1_score, train_loss, valid_loss)
        )


if __name__ == '__main__':
    main()