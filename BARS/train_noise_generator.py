import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

import sys
import json
import random
import argparse
from tqdm import tqdm
from torch import vmap
from multiprocessing import Pool
from torchvision import transforms

sys.path.append('.') # run in directory: CertTA_public/
from evaluation.utilities import *
from evaluation.data_loader import load_data, flow_preprocessing, dimension_alignment
from certification.smoothing import generate_smoothing_samples_bars, int8_overflow, vocab_overflow
from certification.radius import p_A_lower_confidence_bound, radius_bars

sys.path.append('./BARS')
from BARS.smoothing import Noise
from BARS.distribution_transformer import distribution_transformers, loss_functions
from BARS.optimizing_noise import AverageMeter, calc_mean_robust_radius

sys.path.append('./model/Kitsune')
from model.Kitsune.KitNET import KitNET

sys.path.append('./model/DF')
from model.DF.DFNet import DFNet

sys.path.append('./model/YaTC')
import model.YaTC.models_YaTC as models_YaTC

sys.path.append('./model/TrafficFormer')
from model.TrafficFormer.fine_tuning.run_classifier import Classifier
from model.TrafficFormer.uer.utils import *


def bars_opts(parser):
    parser.add_argument("--feature_noise_distribution", default="gaussian", 
                        choices=["gaussian", "isru", "isru_gaussian", "isru_gaussian_arctan"]) # Feature noise distribution
    parser.add_argument("--learning_rate_shape", type=float, default=1e-2) # Learning rate for optimzing noise shape
    parser.add_argument("--nt_shape", type=int, default=10) # Number of noised samples for optimzing noise shape
    parser.add_argument("--lambda_shape", type=float, default=1e-2) # Regularizer weight
    parser.add_argument("--num_epochs_shape", type=int, default=30) # Number of epochs for optimzing noise shape
    parser.add_argument("--n0", type=int, default=100) # Number of noised samples for identify cA
    parser.add_argument("--n", type=int, default=100) # Number of noised samples for estimate pA
    parser.add_argument("--alpha", type=float, default=1e-3) # Failure probability
    parser.add_argument("--init_step_size_scale", type=float, default=5e-2) # Initial update step size of t for optimzing noise scale
    parser.add_argument("--init_ptb_t_scale", type=float, default=1e-2) # Initial perturbation of t for optimzing noise scale
    parser.add_argument("--decay_factor_scale", type=float, default=0.5) # Decay factor for optimzing noise scale
    parser.add_argument("--max_decay_scale", type=int, default=6) # Maximum decay times for optimzing noise scale
    parser.add_argument("--max_iter_scale", type=int, default=100) # Maximum iteration times for optimzing noise scale
    parser.add_argument("--batch_size_iteration_certify", type=int, default=16) # Batch size of certified samples for robustness certification
    parser.add_argument("--batch_size_memory_certify", type=int, default=1000000) # Batch size of noised samples for robustness certification
    parser.add_argument("--print_step_certify", type=int, default=10) # Step size for showing certification progress


def initialize_model(args):
    if args.model == 'TrafficFormer':
        args.train = True
        args = load_hyperparam(args, args.bert_config_path)
        
        args.tokenizer = str2tokenizer[args.tokenizer](args)
        model = Classifier(args)
        load_or_initialize_parameters(model, args.model_dir + 'best_model.bin')
        model = model.to(args.device)
        args.d = args.seq_length
        args.nt_shape = 1
        args.num_epochs_shape = 10
        args.batch_size_iteration_certify = 8
        args.n = 50
    elif args.model == 'YaTC':
        model = models_YaTC.__dict__['TraFormer_YaTC'](
            num_classes=args.labels_num,
            drop_path_rate=args.drop_path,
        )
        load_or_initialize_parameters(model, args.model_dir + 'best_model.bin')
        model = model.to(args.device)
        args.d = (args.header_bytes + args.payload_bytes) * args.input_packets
        args.nt_shape = 3
        args.num_epochs_shape = 20
        args.batch_size_iteration_certify = 8
    elif args.model == 'DF':
        INPUT_SHAPE = [2, args.max_flow_length]
        model = DFNet(INPUT_SHAPE, args.labels_num)
        load_or_initialize_parameters(model, args.model_dir + 'best_model.bin')
        model = model.to(args.device)
        args.d = args.max_flow_length * 2
    elif args.model == 'Kitsune':
        model = KitNET(args.feature_num, args.labels_num, args.maxAE, feature_map=[[i for i in range(args.feature_num)]], device=args.device)
        with open(args.model_dir + 'best_model.bin', 'rb') as file:
            model = pickle.load(file)
        model = model.to(args.device)
        args.d = args.feature_num
    elif args.model in ['Whisper', 'kFP']:
        raise Exception('BARS is not applicable because gradient descent can not be performed!')

    return args, model


@torch.no_grad()
def evaluate(args, model, eval_flows):
    if args.model not in ['Whisper', 'kFP']:
        model.eval()
   
    # Data preprocessing
    instances = [] # instances: [(x, y, ...), ...]
    for flow in tqdm(eval_flows, desc='Flow Preprocessing', disable=True):
        instance_flow = flow_preprocessing(flow, args)
        if instance_flow == -1:
            raise Exception('Bad Flow:' + flow['source_file'])
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

        # Forward Propagation
        y_true.extend(input_output_batch[1])
        if args.model == 'TrafficFormer':
            input_output_batch = [torch.LongTensor(io).to(args.device) for io in input_output_batch]
            _, logits = model(input_output_batch[0], input_output_batch[1], input_output_batch[2], None)
        elif args.model == 'YaTC':
            with torch.cuda.amp.autocast():
                src = torch.stack(input_output_batch[0]).to(args.device)
                logits = model(src)
        elif args.model == 'DF':            
            src = torch.tensor(input_output_batch[0]).to(args.device)
            logits = model(src)
        elif args.model == 'Kitsune':
            src = torch.tensor(input_output_batch[0]).to(args.device)
            logits, _ = model(src)

        _, pred = logits.topk(1, 1, True, True)
        y_pred.extend(pred.t()[0].cpu())

    return y_true, y_pred


def create_noise_generator(args):
    dist_trans = distribution_transformers[args.feature_noise_distribution](args.d).to(args.device)
    return Noise(dist_trans, args.d, args.device)


def smoothed_classification(args, instance_flow, model, noise_generator, t, samples_num, alpha, feat_shape):
    torch.cuda.empty_cache()
    
    instances = generate_smoothing_samples_bars(instance_flow, noise_generator, t, args.model, samples_num)
    
    # Batching
    input_output_batch = []
    for _ in range(len(instances[0])):
        input_output_batch.append([])
    for instance in instances:
        for k in range(len(instance)):
            input_output_batch[k].append(instance[k])

    if args.model == 'TrafficFormer':
        input_output_batch = [torch.LongTensor(io).to(args.device) for io in input_output_batch]
        _, logits = model(input_output_batch[0], None, input_output_batch[2], None)
    elif args.model == 'YaTC':
        with torch.cuda.amp.autocast():
            src = torch.stack(input_output_batch[0]).to(args.device)
            logits = model(src)
    elif args.model == 'DF':
        src = torch.tensor(input_output_batch[0]).to(args.device)
        logits = model(src)
    elif args.model == 'Kitsune':
        src = torch.tensor(input_output_batch[0]).to(args.device)
        logits, _ = model(src)

    _, pred = logits.topk(1, 1, True, True)
    class_count = np.zeros([args.labels_num])
    for c in pred:
        class_count[int(c)] += 1
    c_A = np.argmax(class_count)
    p_A = class_count[c_A] / sum(class_count)

    p_A = p_A_lower_confidence_bound(p_A, samples_num, alpha)
    radius_feat_dim, radius_feat = radius_bars(p_A, noise_generator, t)

    return c_A, radius_feat_dim, radius_feat

def train_noise_generator(args, model, flows, certify_class):
    print("***** Filter flows with correct prediction *****")
    _, y_pred = evaluate(args, model, flows)
    train_flows = []
    for i in range(len(flows)):
        if y_pred[i] == certify_class:
            train_flows.append(flows[i])
    train_instances = [] # instances: [(x, y, ...), ...]
    for flow in tqdm(train_flows, desc='Flow Preprocessing', disable=False):
        instance_flow = flow_preprocessing(flow, args)
        instance_flow = dimension_alignment(args, instance_flow)
        if instance_flow == -1:
            raise Exception('Bad Flow:' + flow['source_file'])
        train_instances.append(instance_flow)

    print("***** Start to train the noise generator *****")
    pred_train = certify_class
    classifier = model
    noise_generator = create_noise_generator(args)
    criterion_shape = loss_functions["acid"](args.lambda_shape, certify_class)
    learning_rate_shape = args.learning_rate_shape
    nt_shape = args.nt_shape
    num_epochs_shape = args.num_epochs_shape
    d = args.d
    num_classes = args.labels_num
    n0 = args.n0
    n = args.n
    alpha = args.alpha
    init_step_size_scale = args.init_step_size_scale
    init_ptb_t_scale = args.init_ptb_t_scale
    decay_factor_scale = args.decay_factor_scale
    max_decay_scale = args.max_decay_scale
    max_iter_scale = args.max_iter_scale
    batch_size_iteration = args.batch_size_iteration_certify
    batch_size_memory = args.batch_size_memory_certify
    print_step = args.print_step_certify
    save_dir = args.save_dir
    classifier_device = args.device

    assert classifier_device == noise_generator.device

    print("***** Optimize noise shape *****")
    num_samples_shape = 10000

    random.shuffle(train_instances)
    instances = train_instances[:min(num_samples_shape, len(train_instances))]
    batch_num = math.ceil(len(instances) / batch_size_iteration)

    opt = torch.optim.Adam(noise_generator.distribution_transformer.parameters(), lr=learning_rate_shape)

    noise_norm = noise_generator.sample_norm(nt_shape).repeat(1, batch_size_iteration).view(-1, noise_generator.d)

    loss_record = AverageMeter()
    classifier.train()

    for epoch in range(0, num_epochs_shape + 1):
        for batch_idx in range(batch_num):
            torch.cuda.empty_cache()
            # Batching
            input_output_batch = []
            for _ in range(len(instances[0])):
                input_output_batch.append([])
            for instance_idx in range(batch_idx * batch_size_iteration, min(len(instances), (batch_idx + 1) * batch_size_iteration)):
                instance = instances[instance_idx]
                for k in range(len(instance)):
                    input_output_batch[k].append(instance[k])
            
            # Get input tensor
            if args.model == 'TrafficFormer':
                X = torch.LongTensor(input_output_batch[0])
            elif args.model == 'YaTC':
                X = torch.tensor(input_output_batch[2])
            X = X.to(classifier_device)

            if X.size(0) < batch_size_iteration:
                noise_idx = torch.cat([torch.arange(X.size(0)) + i * batch_size_iteration \
                    for i in range(nt_shape)], 0).to(X.device)
            else:
                noise_idx = torch.arange(noise_norm.size(0)).to(X.device)
            noise_feat = noise_generator.norm_to_feat(noise_norm[noise_idx, :])
            feat_shape = list(X.shape[1:])
            noise_feat = noise_feat.reshape([-1] + feat_shape)
            noised_X = X.repeat([nt_shape] + [1] * len(feat_shape)) + noise_feat

            if args.model == 'TrafficFormer':
                noised_X = vmap(vocab_overflow)(noised_X)
                tgt = torch.LongTensor(input_output_batch[1]).repeat([nt_shape] + [1] * len(feat_shape)).to(classifier_device)
                seg = torch.LongTensor(input_output_batch[2]).repeat([nt_shape] + [1] * len(feat_shape)).to(classifier_device)
                _, score = classifier(noised_X, tgt, seg, None)
            elif args.model == 'YaTC':
                noised_X = vmap(int8_overflow)(noised_X)
                src = []
                for flow_byte_tensor in noised_X:
                    flow_byte_tensor = flow_byte_tensor.unsqueeze(0)
                    mean = [0.5]
                    std = [0.5]
                    transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Normalize(mean, std),
                    ])
                    src_img = transform(flow_byte_tensor)
                    src.append(src_img)
                src = torch.stack(src)
                score = classifier(src)
            elif args.model == 'DF':
                score = classifier(noised_X)
            elif args.model == 'Kitsune':
                score, _ = classifier(noised_X)
            loss = criterion_shape(score, noise_generator.get_weight())

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            
            loss_record.update(loss.item())
            if (batch_idx % print_step == 0) or ((print_step > batch_num) and (batch_idx == batch_num)):
                print("(Class %d) Batch: [%d/%d][%d/%d] | Loss: %.6f" % (certify_class, epoch, num_epochs_shape, batch_idx, batch_num, loss_record.val))

        print("Epoch: [%d/%d] | Loss (Avg): %.6f" % (epoch, num_epochs_shape, loss_record.avg))
        loss_record.reset()

    torch.save(noise_generator.distribution_transformer, os.path.join(save_dir, "checkpoint-distribution-transformer-" + str(criterion_shape.certify_class)))
    w = open(os.path.join(save_dir, "t-" + str(criterion_shape.certify_class)), "w")
    w.write("%.6e" % (1))
    w.close() 
    # return
    
    print("***** Optimize noise scale *****")
    num_samples_scale = 1000

    random.shuffle(train_instances)
    instances = train_instances[:min(num_samples_scale, len(train_instances))]

    classifier.eval()
    ptb_t_scale = init_ptb_t_scale
    step_size_scale = init_step_size_scale
    decay_scale = 0
    iter_scale = 0
    t = 0.0
    grad_sign_last = 1
    torch.set_grad_enabled(False)
    while (iter_scale < max_iter_scale) and (decay_scale < max_decay_scale):
        cA_record = []
        robust_radius_record = []
        for flow_idx in tqdm(range(len(instances)), desc='(Class {}) Calculating mean_robust_radius'.format(certify_class)):
            instance_flow = instances[flow_idx]
            cA, _, robust_radius = smoothed_classification(args, instance_flow, classifier, noise_generator, t, n, alpha, feat_shape)
            cA_record.append(cA)
            robust_radius_record.append(robust_radius)
        
        mean_robust_radius = calc_mean_robust_radius(pred_train, np.array(cA_record), np.array(robust_radius_record))

        cA_record = []
        robust_radius_record = []
        for flow_idx in tqdm(range(len(instances)), desc='(Class {}) Calculating mean_robust_radius_ptb'.format(certify_class)):
            instance_flow = instances[flow_idx]
            cA, _, robust_radius = smoothed_classification(args, instance_flow, classifier, noise_generator, t + ptb_t_scale, n, alpha, feat_shape)
            cA_record.append(cA)
            robust_radius_record.append(robust_radius)
        
        mean_robust_radius_ptb = calc_mean_robust_radius(pred_train, np.array(cA_record), np.array(robust_radius_record))

        grad_t = (mean_robust_radius_ptb - mean_robust_radius) / ptb_t_scale
        grad_t_sign = 1 if grad_t >= 0 else -1
        if grad_t_sign != grad_sign_last:
            ptb_t_scale *= decay_factor_scale
            step_size_scale *= decay_factor_scale
            grad_sign_last = grad_t_sign
            decay_scale += 1
        t = t + step_size_scale * grad_t_sign

        cA_record = []
        robust_radius_record = []
        for flow_idx in tqdm(range(len(instances)), desc='(Class {}) Calculating mean_robust_radius_last'.format(certify_class)):
            instance_flow = instances[flow_idx]
            cA, _, robust_radius = smoothed_classification(args, instance_flow, classifier, noise_generator, t, n, alpha, feat_shape)
            cA_record.append(cA)
            robust_radius_record.append(robust_radius)
        
        mean_robust_radius_last = calc_mean_robust_radius(pred_train, np.array(cA_record), np.array(robust_radius_record))

        iter_scale += 1
        print("(Class %d) Iter: [%d] | t: %.6e | Robust radius: %.6e | Step size: %.6e | Grad sign: %d" % \
            (certify_class, iter_scale, t, mean_robust_radius_last, step_size_scale, grad_sign_last))
        
        w = open(os.path.join(save_dir, "t-" + str(criterion_shape.certify_class)), "w")
        w.write("%.6e" % (t))
        w.close()     

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()              


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model", default="YaTC", choices=['Kitsune', 'DF', 'YaTC', 'TrafficFormer'])
    bars_opts(parser)
    args = parser.parse_args()
    
    print('Training BARS for the {} model.'.format(args.model))

    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './evaluation/config/{}_{}_config.json'.format(args.model, args.dataset))

    args.augment = None
    args.model_dir = './model/{}/save/{}/{}/'.format(args.model, args.dataset, model_name_generator(args))
    args.save_dir = './model/{}/save/{}/BARS/'.format(args.model, args.dataset)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print('model dir: {}, save dir: {}'.format(args.model_dir, args.save_dir))
    print('--------------------------------------')

    print('Loading the dataset.')
    args.dataset_dir = './dataset/{}/json/'.format(args.dataset)
    with open(args.dataset_dir + 'statistics.json') as fp:
        statistics_json = json.load(fp)
    args.labels_num = statistics_json['label_num']
    args.pcap_level = args.model in ['TrafficFormer', 'YaTC']
    args.max_flow_length = {'CICDOH20': 100, 'TIISSRC23': 100}[args.dataset]
    
    train_flows = load_data(args.dataset_dir + 'train.json', args.pcap_level, args.max_flow_length)
    args.train_flows_num = len(train_flows)

    print('Building the traffic analysis model.')
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args, model = initialize_model(args)
    if args.model in ['Kitsune']:
        normalizer = Normalizer()
        normalizer.norm_max = np.load(args.model_dir + "norm_max.npy")
        normalizer.norm_min = np.load(args.model_dir + "norm_min.npy")
        args.normalizer = normalizer

    # train_noise_generator(args, model, train_flows, 0)
    print('build process pool...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    pool = Pool(processes=args.labels_num)
    for i in range(0, args.labels_num):
        pool.apply_async(train_noise_generator, (args, model, train_flows, i), error_callback=print_error)
    print('build finish...')
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()