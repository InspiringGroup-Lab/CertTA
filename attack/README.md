# Generate Adversarial Flows

## 1. Blanket

### Train the substitute DF model for black-box attack

```
python evaluation/train.py --dataset DATASET_NAME --model DF
```

* Choose DATASET_NAME from CICDOH20 and TIISSRC23.

### Train the Blanket attack model

```
python attack/Blanket/train.py --dataset DATASET_NAME --attack_beta_length 200 --attack_beta_time_ms 40 --attack_pr_sel 0.1 --attack_r_additive_star 200 --attack_insert_pkts 2
```

* Specify the attack intensities by setting attack-related paramters (i.e., `attack_beta_length`, `attack_beta_time_ms`, `attack_pr_sel`,  `attack_r_additive_star` and `attack_insert_pkts`).

* The trained model and training logs will be saved in the `/CertTA_public/attack/Blanket/DATASET_NAME/` directory.

### Generate adversarial perturbations on test flows

```
python attack/Blanket/generate_attack_actions.py --dataset DATASET_NAME --attack_beta_length 200 --attack_beta_time_ms 40 --attack_pr_sel 0.1 --attack_r_additive_star 200 --attack_insert_pkts 2
```

* The adversarial perturbations (including packet length padding, timing delays and packet insertion) on test flows will be saved in a `attack.json` file in the same directory of the trained attack model.

## 2. Amoeba

### Prepare the environment

```
pip install -r attack/Amoeba/requirements.txt
```

### Preprocess the dataset

```
python attack/Amoeba/preprocess_dataset.py
```

* This step will preprocess the dataset and save the processed data in the `/CertTA_public/attack/Amoeba/dataset/` directory.

### Switch to the Amoeba directory

```
cd attack/Amoeba
```

### Train the Amoeba attack model

```
python src/train_amoeba.py --dataset DATASET_NAME
```

* Based on reinforcement learning, Amoeba adaptively trains its attack model to evade the classification of a CertTA-certified DF model.

* The trained attack model will be saved in the `/CertTA_public/attack/Amoeba/saved_models/` directory.

### Generate adversarial perturbations on test flows

```
python src/generate_attack.py --dataset DATASET_NAME --attack_beta_length 200 --attack_beta_time_ms 40 --attack_pr_sel 0.1 --attack_r_additive_star 200 --attack_insert_pkts 2
```

* Specify the attack intensities by setting attack-related paramters (i.e., `attack_beta_length`, `attack_beta_time_ms`, `attack_pr_sel`,  `attack_r_additive_star` and `attack_insert_pkts`).

* The adversarial perturbations (including packet length padding, timing delays and packet insertion) on test flows will be saved in a `attack.json` file in the `/CertTA_public/attack/Amoeba/DATASET_NAME/` directory.

## 3. Prism

```
python attack/Prism/run_prism.py --dataset DATASET_NAME --attack_beta_length 200 --attack_beta_time_ms 40 --attack_pr_sel 0.1 --attack_r_additive_star 200 --attack_insert_pkts 2
```
* Specify the attack intensities by setting attack-related paramters (i.e., `attack_beta_length`, `attack_beta_time_ms`, `attack_pr_sel`,  `attack_r_additive_star` and `attack_insert_pkts`).
  
* The adversarial perturbations (including packet length padding, timing delays and packet insertion) on test flows will be saved in a `attack.json` file in the `/CertTA_public/attack/Prism/DATASET_NAME/` directory.