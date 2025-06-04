<div align="center">

<h1 >CertTA: Certified Robustness Made Practical <br>for Learning-Based Traffic Analysis</h1>

[![License](https://img.shields.io/github/license/CBackyx/RingSG)](https://opensource.org/licenses/MIT)
<!-- [![ePrint]()]() -->

</div>

> This repository contains the artifacts of paper **CertTA: Certified Robustness Made Practical for Learning-Based Traffic Analysis**, which has been accepeted by USENIX Security 2025. A [Zenodo repository](https://doi.org/10.5281/zenodo.15580292) is hosted synchronously to facilitate the use of these artifacts.

# Repository Overview

Our artifact includes the following directories: 

* [dataset/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/dataset) contains the CICDOH20 and TIISSRC23 datasets, including the processed json files and original PCAP files of flow samples.

* [model/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/model) contains the implementations of six supervised traffic analysis systems (i.e., kFP, Kitsune (supervised), Whisper (supervised), DFNet, YaTC and TrafficFormer) and three unsupervised anomaly detection systems (i.e., KMeans, Kitsune, Whisper).

* [certification/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/certification) contains the implementations of CertTA's multi-modal smoothing mechanism and the functions for solving CertTA's robustness region against multi-modal adversarial perturbations. 

* [attack/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/attack) contains the implementations of three multi-modal adversarial attacks (i.e., Blanket, Amoeba, Prism).

* [BARS/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/BARS) contains the implementations of a baseline certification method BARS.

* [evaluation/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/evaluation) contains the source codes for training and evaluating certified traffic analysis models. Our framework supports both CertTA and baseline certification methods (i.e., VRS, BARS and RS-Del) for building certified traffic analysis models.

* [integration/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/integration) contains the source codes for building and evaluating the integrated system of anomaly detectors and certified traffic analysis models.

> Due to the repository size limit, the PCAP files of traffic datasets and the pretrained checkpoints of the YaTC and TrafficFormer models are not provided in our [Github repository](https://github.com/InspiringGroup-Lab/CertTA). The complete artifiacts can be accessed in our [Zenodo repository](https://doi.org/10.5281/zenodo.15580292).

## Environment Setup

To ensure the proper functioning of our artifacts, please follow the commands below:

1. Ensure that you have `conda` installed on your system. If you do not have `conda`, you can install it as part of the Anaconda distribution or Miniconda.
2. Open a terminal or command prompt.
3. Create a new conda environment with the name of your choice (e.g., `CertTA`) and intall all the required packages listed in `environment.yml`:
   
   ```bash
   conda create -n CertTA -f environment.yml
   ```
4. Once the environment is created, activate it by running:
   
   ```bash
   conda activate CertTA
   ```
   This will switch your command line environment to use the newly created conda environment with all the necessary packages.

> This implementation has been successfully tested in **Ubuntu 20.04** server with **Python 3.8.18**.

# Step-by-Step Instructions

* In [evaluation/README.md](https://github.com/InspiringGroup-Lab/CertTA/tree/main/evaluation#readme), we privide step-by-step instructions of implementing the original traffic analysis models and the certified traffic anlaysis models with different certification methods.

* In [integration/README.md](https://github.com/InspiringGroup-Lab/CertTA/tree/main/integration#readme), we privide step-by-step instructions of implementing the integrated system of anomaly detectors and certified traffic analysis models.

* In [attack/README.md](https://github.com/InspiringGroup-Lab/CertTA/tree/main/attack#readme), we privide step-by-step instructions of generating adversarial flows based on different attack methods.


> Unless otherwise specified, the codes should be run in the root directory of this repository (i.e., the `CertTA_public` directory) to properly import dependency files. Please post an issue in our [Github repository](https://github.com/InspiringGroup-Lab/CertTA) or send an email to [yanjz22@mails.tsinghua.edu.cn](mailto:yanjz22@mails.tsinghua.edu.cn) if you have any questions. Have fun!

# Credit

Cite our paper as follows if you find this code repository is useful to you. 

@inproceedings{yan2025certta,  title={{CertTA: Certified Robustness Made Practical for Learning-Based Traffic Analysis}}, author={Yan, Jinzhu and Liu, Zhuotao and Yuyang Xie and Shiyu Liang and Lin Liu and Ke Xu}, booktitle={34th USENIX Security Symposium}, year={2025}}

> The implementations of traffic analysis models ([kFP](https://github.com/jhayes14/k-FP), [Kitsune](https://github.com/ymirsky/Kitsune-py/tree/master), [Whisper](https://github.com/fuchuanpu/Whisper), [DFNet](https://github.com/deep-fingerprinting/df), [YaTC](https://github.com/NSSL-SJTU/YaTC), [TrafficFormer](https://github.com/IDP-code/TrafficFormer)), baseline certification methods ([VRS](https://github.com/locuslab/smoothing), [BARS](https://github.com/KaiWangGitHub/BARS), [RS-Del](https://github.com/Dovermore/randomized-deletion)) and adversarial attacks ([Blanket](https://github.com/SPIN-UMass/BLANKET), [Amoeba](https://github.com/Mobile-Intelligence-Lab/Amoeba), [Prism](https://github.com/SecTeamPolaris/Prism)) are based on their open-source repositories. The original PCAP files of the [CICDOH20](https://www.unb.ca/cic/datasets/dohbrw-2020.html) and [TIISSRC23](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23) datasets are obtained from their open-source websites. Many thanks to the authors.