def smoothing_opts(parser):
    parser.add_argument("--samples_num", type=int, default=1000, 
                            help="the number of smoothing samples")
    parser.add_argument("--alpha", type=float, default=1e-3, 
                            help="failure probability of certification based on randomized smoothing")
    
    # CertTA
    parser.add_argument("--beta_time_ms", type=int, default=None, 
                            help="the expectation of the exponential perturbation added to the timing of each packet.") # packet timing delay
    parser.add_argument("--beta_length", type=int, default=None, 
                            help="the expectation of the exponential perturbation added to the length of each packet.") # packet length padding 
    parser.add_argument("--pr_sel", type=float, default=None, 
                            help="the proportion of selected packets in a flow") # packet selection
    
    # VRS
    parser.add_argument("--sigma_vrs", type=float, default=None)


def attack_opts(parser):
    parser.add_argument("--attack", type=str, default=None, choices=['Blanket', 'Amoeba', 'Prism'])

    # Attack Intensities
    parser.add_argument("--attack_beta_length", type=int, default=100)
    parser.add_argument("--attack_beta_time_ms", type=int, default=40)
    parser.add_argument("--attack_pr_sel", type=float, default=0.15)
    parser.add_argument("--attack_r_additive_star", type=float, default=21.958)
    parser.add_argument("--attack_insert_pkts", type=int, default=2)
    
    
def training_opts(parser):
    parser.add_argument("--optimizer", type=str, default="adamW")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_num", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.01)