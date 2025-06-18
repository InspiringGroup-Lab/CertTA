import os
import sys
import json
import argparse

sys.path.append('.') # run in directory: CertTA_public/
from certification.radius import *
from evaluation.utilities import *
from evaluation.opts import smoothing_opts, attack_opts, training_opts


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model_AD", default="Kitsune", choices=['KMeans', 'Whisper', 'Kitsune'])
    parser.add_argument("--FPR_threshold", type=float, default=0.01)
    parser.add_argument("--model", default="YaTC", choices=['kFP', 'Kitsune', 'Whisper', 'DF', 'TrafficFormer', 'YaTC'])
    parser.add_argument("--augment", type=str, default='CertTA', choices=['CertTA', 'VRS', 'BARS', 'RSDel'],
                        help='train with the smoothing samples (perturbed flows)')
    parser.add_argument("--smoothed", type=str, default='CertTA', choices=['CertTA', 'VRS', 'BARS', 'RSDel'],
                        help='test with randomized smoothing')
    parser.add_argument("--truncate", type=float, default=None, choices=[None, 0.25, 0.5, 0.75])
    smoothing_opts(parser)
    attack_opts(parser)
    training_opts(parser)
    args = parser.parse_args()
    print('Testing the integration of the {}_AD model and the {}-certified {} model.'.format(args.model_AD, args.smoothed, args.model))

    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './integration/config/{}_AD_{}_config.json'.format(args.model_AD, args.dataset))
    args = load_hyperparam(args, './evaluation/config/{}_{}_config.json'.format(args.model, args.dataset))
    
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
    print('--------------------------------------')
    
    args.dataset_dir = './dataset/{}/json/'.format(args.dataset)
    with open(args.dataset_dir + 'statistics.json') as fp:
        statistics_json = json.load(fp)
    args.labels_num = statistics_json['label_num']
    
    y_true = []
    y_pred = []
    for parent, _, files in os.walk(args.result_dir):
        for json_file in files:
            if json_file.find('.json') == -1:
                continue
            with open(os.path.join(parent, json_file)) as fp:
                results = json.load(fp)
            for res in results:
                y_true.append(res['label'])
                y_pred.append(res['c_A'])
    
    print('Calculating the defense success rate.')
    success = [0] * args.labels_num
    fail = [0] * args.labels_num
    for i in range(len(y_true)):
        if y_pred[i] == -1 and args.attack is not None:
            success[y_true[i]] += 1
        elif y_pred[i] == y_true[i]:
            success[y_true[i]] += 1
        else:
            fail[y_true[i]] += 1
    
    success_rate = [success[i] / (success[i] + fail[i]) if (success[i] + fail[i]) > 0 else 0 for i in range(args.labels_num)]
    with open(args.result_dir + 'defense_success_rate.txt', 'w') as fp:
        fp.writelines("| Defense Success Rate {:6.3f}, Macro {:6.3f}\n".format(sum(success) / len(y_true), sum(success_rate) / len(success_rate)))

        for true_label in range(args.labels_num):
            samples_num = success[true_label] + fail[true_label]
            line = "| true label {:4d} | samples num {:6d} | success rate {:6.3f} | fail rate {:6.3f} |".format(
                    true_label, int(samples_num), success[true_label] / samples_num if samples_num > 0 else 0, fail[true_label] / samples_num if samples_num > 0 else 0)
            fp.writelines(line + " |\n")


if __name__ == '__main__':
    main()