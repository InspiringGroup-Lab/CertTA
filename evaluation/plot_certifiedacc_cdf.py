import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
font_size = 23
line_width = 5
marker_size = 10
title_size = 22

sys.path.append('.') # run in directory: CertTA_public/
from certification.radius import *
from evaluation.utilities import *
from evaluation.opts import smoothing_opts


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset", default="CICDOH20", choices=['CICDOH20', 'TIISSRC23'])
    parser.add_argument("--model", default="DF", choices=['kFP', 'Kitsune', 'Whisper', 'DF', 'TrafficFormer', 'YaTC'])
    parser.add_argument("--augment", type=str, default='CertTA', choices=['CertTA'],
                        help='train with the smoothing samples (perturbed flows)')
    parser.add_argument("--smoothed", type=str, default='CertTA', choices=['CertTA'],
                        help='test with randomized smoothing')
    smoothing_opts(parser)
    args = parser.parse_args()
    print('Plotting the CDF of certified accuracy for the {} model.'.format(args.model))

    print('Loading the model hyperparameters from the config file.')
    args = load_hyperparam(args, './evaluation/config/{}_{}_config.json'.format(args.model, args.dataset))

    args.smoothing_params = {
        'beta_length': args.beta_length,
        'beta_time_ms': args.beta_time_ms,
        'pr_sel': args.pr_sel
    }

    args.save_dir = './model/{}/save/{}/{}/'.format(args.model, args.dataset, model_name_generator(args))
    print('save dir: {}'.format(args.save_dir))
    args.result_dir = args.save_dir + ('base/' if args.smoothed is None else '{}/'.format(args.smoothed))
    print('smoothed:', args.smoothed)
    args.result_dir += 'clean/'
    print('clean test set')
    print('--------------------------------------')

    # Load the p_A of correctly classified clean samples for radius calculation
    test_flow_num = 0
    correct_clean_results = []
    clean_result_dir = args.save_dir + args.smoothed + '/clean/'
    for parent, _, files in os.walk(clean_result_dir):
        for json_file in files:
            if json_file.find('.json') == -1:
                continue
            with open(os.path.join(parent, json_file)) as fp:
                clean_results = json.load(fp)
            for res in clean_results:
                test_flow_num += 1
                if res['label'] == res['c_A']:
                    correct_clean_results.append(res)
    correct_num = len(correct_clean_results)
    
    n_ins_set = [i for i in range(6)]
    r_additive_star_lists = [[0] * correct_num for _ in range(len(n_ins_set))]
    for idx, res in enumerate(correct_clean_results):
        n = res['packet_num']
        p_A = p_A_lower_confidence_bound(res['p_A'], args.samples_num, args.alpha)
        if args.pr_sel is not None:
            d = int(np.ceil(args.pr_sel * n))
        else:
            d = n
        S_jnt = radius_joint_exp(p_A, d, args.beta_length, args.beta_time_ms, n)
        for point in S_jnt:
            n_del, n_ins, n_sub, r_additive_star = point
            if n_del == 0 and n_ins in n_ins_set and n_sub == 0:
                r_additive_star_lists[n_ins][idx] = r_additive_star
    
    color = ['#3F8AF7', '#7F3DC5', '#56858B', '#5761AE', '#CE5030', '#EA395D']
    linestyle = ['-', '--', '-.', (0, (5, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1, 1, 1)), ':']
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(1, 1, 1)
    
    intensities_at_certified_acc = {}
    for certified_acc in [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        intensities_at_certified_acc[str(certified_acc)] = []
    
    legend_handle = []
    max_x = max(r_additive_star_lists[0]) + 10
    for n_ins in n_ins_set:
        r_additive_star_list = r_additive_star_lists[n_ins]
        x = [max_x * i / 1000 for i in range(1000)]
        y = [sum(np.array(r_additive_star_list) > i) / test_flow_num for i in x]
        curve, = ax.plot(x, y, color=color[n_ins], linewidth=line_width, linestyle=linestyle[n_ins])
        legend_handle.append(curve)
        
        for certified_acc in intensities_at_certified_acc.keys():
            for i in range(len(y)):
                if y[i] < float(certified_acc):
                    if i > 0:
                        intensities_at_certified_acc[certified_acc].append((n_ins, x[i-1]))
                    break
    
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.ylim(-0.03, 1.05)
    ax.set_title(args.model, {'size': font_size}, pad=5)
    ax.set_xlabel('Robustness Radius ' + r'$(r^{add}_*@n^{ins})$', fontsize=font_size, labelpad=5)
    ax.set_ylabel('Certified Accuracy', fontsize=font_size, labelpad=10)
    ax.grid()
    
    ax.legend(handles=legend_handle, labels=[r'$n_{ins} = $' + str(i) for i in n_ins_set],
        loc=0,
        # ncol=5,
        prop={'size': font_size},
        frameon=True, labelspacing=0.5, columnspacing=2, handletextpad=0.5, handlelength=3)
    plt.setp(ax.spines.values(), linewidth=3)
    
    plt.show()
    fig.savefig(args.result_dir + 'certifiedacc_cdf.pdf', bbox_inches='tight', dpi=1000)
    with open(args.result_dir + 'attack_intensities_at_certifiedacc', 'w') as fp:
        intensities_at_certified_acc['certified_acc'] = '[(n_ins, r_additive_star), ...]'
        json.dump(intensities_at_certified_acc, fp, indent=1)

    
if __name__ == '__main__':
    main()