# CertTA

The repository of paper **CertTA: Certified Robustness Made Practical for Learning-Based Traffic Analysis**, which has been accepeted by USENIX Security 2025. 

# Repository Overview

Our artifact includes the following directories: 

* [dataset/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/dataset) contains the CICDOH20 and TIISSRC23 datasets, including the processed json files and original PCAP files of flow samples.

* [model/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/model) contains the implementations of six supervised traffic analysis systems (i.e., kFP, Kitsune (supervised), Whisper (supervised), DFNet, YaTC and TrafficFormer) and three unsupervised anomaly detection systems (i.e., KMeans, Kitsune, Whisper).

* [certification/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/certification) contains the implementations of CertTA's multi-modal smoothing mechanism and the functions for solving CertTA's robustness region against multi-modal adversarial perturbations. 

* [attack/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/attack) contains the implementations of three multi-modal adversarial attacks (i.e., Blanket, Amoeba, Prism).

* [BARS/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/BARS) contains the implementations of a baseline certification method BARS.

* [evaluation/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/evaluation) contains the source codes for training and evaluating certified traffic analysis models. Our framework supports both CertTA and baseline certification methods (i.e., VRS, BARS and RS-Del) for building certified traffic analysis models.

* [integration/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/integration) contains the source codes for building and evaluating the integrated system of anomaly detectors and certified traffic analysis models.

Follow the detailed instructions in each directory to implement the original traffic analysis models, traffic anlaysis models with different certification methods, integrated systems and adversarial attacks for evaluation.

Note:

* The software environment is provided in `environment.yml`. 

* Unless otherwise specified, the codes should be run in the root directory of this repository to properly import dependency files.

* Due to the repository size limit, the PCAP files of traffic datasets and the pretrained checkpoints of the YaTC and TrafficFormer models are not provided in our [Github repository](https://github.com/InspiringGroup-Lab/CertTA). The complete artifiacts can be accessed in our [Zenodo repository](https://doi.org/10.5281/zenodo.15580293).

# Contact

Please post an issue in our [Github repository](https://github.com/InspiringGroup-Lab/CertTA) or send an email to [yanjz22@mails.tsinghua.edu.cn](yanjz22@mails.tsinghua.edu.cn) if you have any questions.

# Credit

Cite our paper as follows if you find this code repository is useful to you. 

@inproceedings{yan2025certta,  title={{CertTA: Certified Robustness Made Practical for Learning-Based Traffic Analysis}}, author={Yan, Jinzhu and Liu, Zhuotao and Yuyang Xie and Shiyu Liang and Lin Liu and Ke Xu}, booktitle={34th USENIX Security Symposium}, year={2025}}

Note:
*  The implementations of traffic analysis models ([kFP](https://github.com/jhayes14/k-FP), [Kitsune](https://github.com/ymirsky/Kitsune-py/tree/master), [Whisper](https://github.com/fuchuanpu/Whisper), [DFNet](https://github.com/deep-fingerprinting/df), [YaTC](https://github.com/NSSL-SJTU/YaTC), [TrafficFormer](https://github.com/IDP-code/TrafficFormer)), baseline certification methods ([VRS](https://github.com/locuslab/smoothing), [BARS](https://github.com/KaiWangGitHub/BARS), [RS-Del](https://github.com/Dovermore/randomized-deletion)) and adversarial attacks ([Blanket](https://github.com/SPIN-UMass/BLANKET), [Amoeba](https://github.com/Mobile-Intelligence-Lab/Amoeba), [Prism](https://github.com/SecTeamPolaris/Prism)) are based on their open-source repositories. 

* The original PCAP files of the [CICDOH20](https://www.unb.ca/cic/datasets/dohbrw-2020.html) and [TIISSRC23](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23) datasets are obtained from their open-source websites.

Many thanks to the authors.