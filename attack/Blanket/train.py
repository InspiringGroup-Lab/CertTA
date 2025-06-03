import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import sys
import json
import time
import torch
import random
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.append('.')
from evaluation.utilities import *
from attack.Blanket.Noiser import *
from attack.Blanket.Decider import *

sys.path.append('./model/DF')
from model.DF.DFNet import DFNet


def load_data(data_file, max_flow_length=100):
    flows = []
    with open(data_file) as fp:
        data = json.load(fp)
    for flow in data:
        assert flow['packet_num'] == len(flow['timestamp'])
        assert len(flow['timestamp']) == len(flow['direction_length'])
        assert flow['timestamp'][0] == 0

        direction_length = flow['direction_length']
        timestamp_ms = [i * 1e3 for i in flow['timestamp']]
        iat_ms = [0] + [timestamp_ms[i] - timestamp_ms[i-1] for i in range(1, flow['packet_num'])]
        packet_num = flow['packet_num']
        LENGTH = max_flow_length
        if packet_num < LENGTH:
            direction_length += [0] * (LENGTH - packet_num)
            iat_ms += [0] * (LENGTH - packet_num)
        else:
            direction_length = direction_length[:LENGTH]
            iat_ms = iat_ms[:LENGTH]
        src_seq = [direction_length, iat_ms]
        tgt = flow['label']
        flows.append((src_seq, tgt))
    return flows


def train_one_epoch(args, model, add_noiser, add_optimizer, insert_noiser, insert_optimizer, decider, train_flows, epoch_idx):
    add_noiser.train()
    add_optimizer.zero_grad()
    insert_noiser.train()
    insert_optimizer.zero_grad()
    
    train_loss = 0
    time_start = time.time()

    instances = train_flows
    random.shuffle(instances)
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

        y_true.extend(input_output_batch[1])
        
        src = torch.tensor(input_output_batch[0]).to(args.device)
        tgt = torch.tensor(input_output_batch[1]).to(args.device)

        add_size_noise, add_delay_noise_ms = add_noiser(src)
        insert_where_noise, insert_size_noise, insert_delay_noise_ms = insert_noiser(src)
        src = decider(src, 
            add_size_noise, add_delay_noise_ms, args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_sel, args.attack_r_additive_star,
            insert_where_noise, insert_size_noise, insert_delay_noise_ms, args.attack_insert_pkts)
        
        logits = model(src)

        # maximize the loss (rather than minimize the loss)
        loss = - nn.CrossEntropyLoss()(logits, tgt)
        loss.backward()
        
        add_optimizer.step()
        add_optimizer.zero_grad()
        insert_optimizer.step()
        insert_optimizer.zero_grad()

        train_loss += loss.detach().item()
        _, pred = logits.topk(1, 1, True, True)
        y_pred.extend(pred.t()[0].cpu())
    
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')

    print('Train_loss', train_loss, 'Precision', precision, 'Recall', recall, 'F1', f1score, 'Cost Time', time.time() - time_start)
    print(confusion_matrix(y_true, y_pred))

    return f1score, train_loss


@torch.no_grad()
def evaluate(args, model, add_noiser, insert_noiser, decider, valid_flows, epoch_idx):
    add_noiser.eval()
    insert_noiser.eval()
    
    eval_loss = 0
    time_start = time.time()

    instances = valid_flows
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

        y_true.extend(input_output_batch[1])
        
        src = torch.tensor(input_output_batch[0]).to(args.device)
        tgt = torch.tensor(input_output_batch[1]).to(args.device)
        
        add_size_noise, add_delay_noise_ms = add_noiser(src)
        insert_where_noise, insert_size_noise, insert_delay_noise_ms = insert_noiser(src)
        src = decider(src, 
            add_size_noise, add_delay_noise_ms, args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_sel, args.attack_r_additive_star,
            insert_where_noise, insert_size_noise, insert_delay_noise_ms, args.attack_insert_pkts)
        
        logits = model(src)

        # maximize the loss (rather than minimize the loss)
        loss = - nn.CrossEntropyLoss()(logits, tgt)

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
    
    # attack parameters
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--attack_beta_length", type=float, default=100)
    parser.add_argument("--attack_beta_time_ms", type=float, default=40)
    parser.add_argument("--attack_pr_sel", type=float, default=0.15)
    parser.add_argument("--attack_r_additive_star", type=float, default=21.958)
    parser.add_argument("--attack_insert_pkts", type=int, default=2)
    
    # training parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_num", type=int, default=50)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--add_learning_rate", type=float, default=5e-3)
    parser.add_argument("--insert_learning_rate", type=float, default=5e-3)
    
    args = parser.parse_args()
    args.model = 'DF'
    print('Training the blanket attack model. (substitute model: {})'.format(args.model))

    args.attack = 'Blanket'
    args.save_dir = './attack/{}/{}/'.format(args.attack, args.dataset)
    args.save_dir += 'Blanket_beta_length_{}_beta_time_ms_{}_pr_sel_{}_r_additive_star_{}_insert_pkts_{}/'.format(
        args.attack_beta_length, args.attack_beta_time_ms, args.attack_pr_sel, args.attack_r_additive_star, args.attack_insert_pkts)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print('save dir: {}'.format(args.save_dir))
    print('--------------------------------------')
    
    print('Loading the dataset.')
    args.dataset_dir = './dataset/{}/json/'.format(args.dataset)
    with open(args.dataset_dir + 'statistics.json') as fp:
        statistics_json = json.load(fp)
    args.labels_num = statistics_json['label_num']
    args.pcap_level = False
    args.max_flow_length = {'CICDOH20': 100, 'TIISSRC23': 100}[args.dataset]
    
    train_flows = load_data(args.dataset_dir + 'train.json', args.max_flow_length)
    valid_flows = load_data(args.dataset_dir + 'valid.json', args.max_flow_length)
    args.train_flows_num = len(train_flows)
    
    print('Loading the substitute model: {}.'.format(args.model))
    args.model_dir = './model/{}/save/{}/{}/'.format(args.model, args.dataset, args.model)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    INPUT_SHAPE = [2, args.max_flow_length]
    model = DFNet(INPUT_SHAPE, args.labels_num)
    load_or_initialize_parameters(model, args.model_dir + 'best_model.bin')
    model = model.to(args.device)
    model.eval()
    model.zero_grad()
    print(torch.cuda.get_device_name(args.device))

    print('Building the noiser models for Blanket.')
    add_noiser = AddNoiser(args.max_flow_length, args.hidden_dim).to(args.device)
    add_optimizer = torch.optim.AdamW(add_noiser.parameters(), lr=args.add_learning_rate)
    insert_noiser = InsertNoiser(args.max_flow_length, args.hidden_dim).to(args.device)
    insert_optimizer = torch.optim.AdamW(insert_noiser.parameters(), lr=args.insert_learning_rate)
    decider = get_decider()
    
    print('Training the noiser models.')
    best_round = 0
    best_valid_f1_score = 1.1
    for epoch in tqdm(range(1, args.epochs_num + 1), desc='Training Epoch'):
        train_f1_score, train_loss = train_one_epoch(args, model, 
            add_noiser, add_optimizer, insert_noiser, insert_optimizer, decider,
            train_flows, epoch
        )

        valid_f1_score, valid_loss = evaluate(args, model, 
            add_noiser, insert_noiser, decider,
            valid_flows, epoch
        )

        if valid_f1_score < best_valid_f1_score:
            best_valid_f1_score = valid_f1_score
            best_round = epoch
            save_model('Blanket', add_noiser, args.save_dir + 'add_noiser.bin')
            save_model('Blanket', insert_noiser, args.save_dir + 'insert_noiser.bin')          
        elif epoch - best_round >= args.early_stop:
            print("Early Stopping!")
            logging(args.save_dir + 'training_log.txt', 'Early Stopping!\n')
            break

        logging(args.save_dir + 'training_log.txt', 
                'Epoch {}, train f1 {}, valid f1 {}, best valid f1 {}, train loss {}, valid loss {}.\n'.format(
                    epoch, train_f1_score, valid_f1_score, best_valid_f1_score, train_loss, valid_loss)
        )


if __name__ == '__main__':
    main()