import os
import sys
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.append('.') # run in directory: CertTA_public/
from certification.radius import *
from evaluation.utilities import *
from evaluation.opts import smoothing_opts, attack_opts, training_opts
from BARS.train_noise_generator import create_noise_generator


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model", default="DF", choices=['kFP', 'Kitsune', 'Whisper', 'DF', 'YaTC', 'TrafficFormer'])
    parser.add_argument("--augment", type=str, default=None, choices=['CertTA', 'VRS', 'BARS'],
                        help='train with the smoothing samples (perturbed flows)')
    parser.add_argument("--smoothed", type=str, default=None, choices=['CertTA', 'VRS', 'BARS'],
                        help='test with randomized smoothing')
    parser.add_argument("--truncate", type=float, default=None, choices=[None, 0.25, 0.5, 0.75])
    smoothing_opts(parser)
    attack_opts(parser)
    training_opts(parser)
    args = parser.parse_args()
    print('Testing the {} model.'.format(args.model))

    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './evaluation/config/{}_{}_config.json'.format(args.model, args.dataset))
    
    args.dataset_dir = './dataset/{}/json/'.format(args.dataset)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.dataset_dir + 'statistics.json') as fp:
        statistics_json = json.load(fp)
    args.labels_num = statistics_json['label_num']
    args.max_flow_length = {'CICDOH20': 100, 'TIISSRC23': 100}[args.dataset]
    
    if args.smoothed == 'CertTA': # Parameters for smoothing samples generation
        args.smoothing_params = {
            'beta_length': args.beta_length,
            'beta_time_ms': args.beta_time_ms,
            'pr_sel': args.pr_sel,
        }
    elif args.smoothed == 'BARS' or args.augment == 'BARS':
        print('Loading BARS noise generators')
        if args.model in ['kFP', 'Whisper']:
            raise Exception('BARS is not applicable because gradient descent can not be performed!')
        elif args.model == 'Kitsune':
            args.d = args.feature_num
        elif args.model == 'DF':
            args.d = args.max_flow_length * 2
        elif args.model == 'YaTC':
            args.d = (args.header_bytes + args.payload_bytes) * args.input_packets
        elif args.model == 'TrafficFormer':
            args.d = args.seq_length

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

    args.save_dir = './model/{}/save/{}/{}{}/'.format(args.model, args.dataset, model_name_generator(args), '_truncate_{}'.format(args.truncate) if args.truncate is not None else '')
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
    print('--------------------------------------')

    y_true = []
    y_pred = []
    correct_results_dict = {}
    for parent, _, files in os.walk(args.result_dir):
        for json_file in files:
            if json_file.find('.json') == -1:
                continue
            with open(os.path.join(parent, json_file)) as fp:
                results = json.load(fp)
            for res in results:
                y_true.append(res['label'])
                y_pred.append(res['c_A'])
                if res['label'] == res['c_A']:
                    correct_results_dict[res['source_file']] = res

    print('Calculating the empirical accuracy.')
    conf_mat = confusion_matrix(y_true, y_pred)
    precisions, recalls, f1s = metric_from_confuse_matrix(conf_mat)
    macro_acc = macro_acc_from_confuse_matrix(conf_mat)
    labels_num = len(precisions)
    with open(args.result_dir + 'empirical_acc.txt', 'w') as fp:
        fp.writelines("| Empirical Accuracy {:6.3f}, Macro {:6.3f}\n".format(len(correct_results_dict.keys()) / len(y_true), macro_acc))
        fp.writelines("| Macro"    
                    "| precision {:6.3f}"
                    "| recall {:6.3f}"
                    "| f1 {:6.3f}\n\n".format(np.mean(precisions), np.mean(recalls), np.mean(f1s)))

        for true_label in range(labels_num):
            samples_num = sum(conf_mat[true_label])
            line = "| true label {:4d} | samples num {:6d} | accuracy {:6.3f} | precision {:6.3f} | recall {:6.3f} | f1 {:6.3f} |".format(
                    true_label, int(samples_num), conf_mat[true_label][true_label] / samples_num, precisions[true_label], recalls[true_label], f1s[true_label])
            for pred_label in range(labels_num):
                line += " {:6d}".format(conf_mat[true_label][pred_label])
            fp.writelines(line + " |\n")
        
        avg_acc = sum([conf_mat[i][i] for i in range(labels_num)]) / len(y_true)
        fp.writelines("| Average Accuracy {:6.3f} |\n".format(avg_acc))
            
    if args.smoothed is not None and args.attack is not None:
        certified_nums = [0] * labels_num
        print('Calculating the certified accuracy.')

        # Load the p_A of correctly classified clean samples for robustness region calculation
        correct_clean_results_dict = {}
        clean_result_dir = args.save_dir + args.smoothed + '/clean/'
        for parent, _, files in os.walk(clean_result_dir):
            for json_file in files:
                if json_file.find('.json') == -1:
                    continue
                with open(os.path.join(parent, json_file)) as fp:
                    clean_results = json.load(fp)
                for res in clean_results:
                    if res['label'] == res['c_A']:
                        correct_clean_results_dict[res['source_file']] = res
        
        # Load the attack samples for distance calculation
        with open(args.attack_set_path) as fp:
            attack_set = json.load(fp)
        for sample in tqdm(attack_set, desc='Certifying'):
            source_file = './dataset/' + sample['source_file'].split('./')[-1]
            try:
                correct_clean_result = correct_clean_results_dict[source_file]
                pred = correct_clean_result['label']
            except:
                continue
            n = correct_clean_result['packet_num']
            p_A = p_A_lower_confidence_bound(correct_clean_result['p_A'], args.samples_num, args.alpha)

            if args.smoothed == 'CertTA':
                # Calculate the perturbation distances between attack sample and clean sample
                if args.pr_sel is not None:
                    d = int(np.ceil(args.pr_sel * n))
                else:
                    d = None
                    
                dis_ins = 0
                adding_actions = []
                for action in sample['actions']:
                    if action['action'] == 'padding':
                        if action['value'] > 0 or action['added_delay'] > 0:
                            adding_actions.append(action)
                    elif action['action'] == 'inserting':
                        dis_ins += 1
                    elif action['action'] == 'inaction':
                        continue
                    else:
                        raise Exception('Unknown attack action: {}!'.format(action['action']))
                
                def custom_sort_key(adding_action):
                    if args.beta_length is not None and args.beta_time_ms is not None:
                        return adding_action['value'] / args.beta_length + adding_action['added_delay'] * 1e3 / args.beta_time_ms
                    elif args.beta_length is not None:
                        return adding_action['value']
                    elif args.beta_time_ms is not None:
                        return adding_action['added_delay'] * 1e3
                    else:
                        return adding_action['value']
                prioritied_adding_actions = sorted(adding_actions, key=custom_sort_key, reverse=True)
                
                dis_points = []
                r_sub = min(radius_packet_substitution(p_A, n, d), n)
                for dis_sub in range(min(r_sub, len(prioritied_adding_actions)) + 1):
                    dis_additive = 0
                    for action in prioritied_adding_actions[dis_sub: dis_sub+d]:
                        beta_length = args.beta_length if args.beta_length is not None else 1e-6
                        beta_time_ms = args.beta_time_ms if args.beta_time_ms is not None else 1e-6
                        dis_additive += (beta_length + beta_time_ms) * (action['value'] / beta_length + action['added_delay'] * 1e3 / beta_time_ms)
                
                    dis_del = 0
                    dis_points.append((dis_del, dis_ins, dis_sub, dis_additive))

                # Calculate certified radius on clean sample
                S_jnt = radius_joint_exp(p_A, d, args.beta_length, args.beta_time_ms, n)

                # Check if the attack sample is within the robustness region of the clean sample
                for dis_point in dis_points:
                    approved = False
                    for surface_point in S_jnt:
                        under_surface = True
                        for i in range(len(surface_point)):
                            if dis_point[i] > surface_point[i]:
                                under_surface = False
                                break
                        if under_surface:
                            approved = True
                            break
                    if approved:
                        certified_nums[pred] += 1
                        break
            elif args.smoothed == 'VRS':
                # Calculate the distances between attack sample and clean sample
                try:
                    input_vector_attack = eval(correct_results_dict[source_file]['input_vector'])
                except:
                    continue
                input_vector_clean = eval(correct_clean_result['input_vector'])
                
                difference = np.array(input_vector_attack) - np.array(input_vector_clean)
                dis_vrs = np.linalg.norm(difference, ord=2)

                # Calculate certified radius on clean sample
                R = radius_gaussian(p_A, args.sigma_vrs)

                # Check if the attack sample is within the robustness region of the clean sample
                if dis_vrs <= R:
                    certified_nums[pred] += 1
            elif args.smoothed == 'BARS':
                # Calculate the distances between attack sample and clean sample
                try:
                    input_vector_attack = eval(correct_results_dict[source_file]['input_vector'])
                except:
                    continue
                input_vector_clean = eval(correct_clean_result['input_vector'])
                
                difference = np.array(input_vector_attack) - np.array(input_vector_clean)

                # Calculate certified radius on clean sample
                radius_norm = norm.ppf(p_A)
                noise_generator = args.noise_generator_cls[pred]
                t = args.t_cls[pred]
                difference = torch.tensor(difference) / t
                difference = difference.reshape(1, -1).to(args.device)
                reverse_difference = noise_generator.distribution_transformer_reverse(difference).cpu().detach().numpy()
                l2_dis_bars = np.linalg.norm(reverse_difference, ord=2)                

                # Check if the attack sample is within the robustness region of the clean sample
                if l2_dis_bars <= radius_norm:
                    certified_nums[pred] += 1
            else:
                raise Exception('Unkonwn certification method!')
        
        certified_acc = [certified_nums[true_label] / sum(conf_mat[true_label]) for true_label in range(labels_num)]
        macro_certified_acc = sum(certified_acc) / labels_num
        with open(args.result_dir + 'certified_acc.txt', 'w') as fp:
            fp.writelines("| Certified Accuracy {:6.3f}, Macro {:6.3f} |\n\n".format(sum(certified_nums) / len(y_true), macro_certified_acc))

            for true_label in range(labels_num):
                samples_num = sum(conf_mat[true_label])
                line = "| true label {:4d} | samples num {:6d} | certified accuracy {:6.3f} |".format(
                        true_label, int(samples_num), certified_nums[true_label] / samples_num)
                fp.writelines(line + "\n")


if __name__ == '__main__':
    main()