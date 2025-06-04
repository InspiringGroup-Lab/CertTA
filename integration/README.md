# Evaluate the Integrted System

### Train Anomaly Detectors

```bash
python integration/train_anomaly_detector.py --dataset DATASET_NAME --model MODEL_NAME
```

* Choose `DATASET_NAME` from CICDOH20 and TIISSRC23. 

* Choose `MODEL_NAME` from three anomaly detection models KMeans, Kitsune and Whisper.

* The trained model and training logs will be saved in the `/CertTA_public/model/MODEL_NAME_AD/save/DATASET_NAME/` directory.

### Evaluate Anomaly Detectors

Follow the instructions in [/CertTA_public/attacks/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/attack) to generate adversarial perturbations based on Blanket, Amoeba and Prism. Then,

```bash
python integration/test_anomaly_detector.py --dataset DATASET_NAME --attack ATTACK_NAME --model MODEL_NAME --FPR_threshold 0.01
```

* Choose ATTACK_NAME from Blanket, Amoeba and Prism. Specify the attack intensities by setting attack-related paramters (e.g., `attack_r_additive_star` and `attack_insert_pkts`) in `/CertTA_public/evaluation/opts.py`. 

* Based on the `FPR_threshold`, we set a threshold from the anomaly detection scores of clean traffic to ensure the upper-bound of the False Positive Rate. 

* The True Positive Rate on the adversarial datasets will be saved as a `anomaly_detection_acc.txt` file n the `/CertTA_public/model/MODEL_NAME_AD/save/DATASET_NAME/` directory.


### Evaluate the Integrated System


Follow the instructions in [/CertTA_public/evaluation/](https://github.com/InspiringGroup-Lab/CertTA/tree/main/evaluation) to train CertTA-certified traffic analysis models. Then,

```bash
python integration/test_integrated_system.py --dataset DATASET_NAME --attack ATTACK_NAME --model_AD MODEL_NAME_AD --FPR_threshold 0.01  --model MODEL_NAME --augment CertTA --smoothed CertTA
```

* Choose ATTACK_NAME from Blanket, Amoeba and Prism. Specify the attack intensities by setting attack-related paramters (e.g., `attack_r_additive_star` and `attack_insert_pkts`) in `/CertTA_public/evaluation/opts.py`. 

* Choose MODEL_NAME_AD from KMeans, Kitsune, Whisper. Based on the `FPR_threshold`, we set a threshold from the anomaly detection scores of clean traffic to ensure the upper-bound of the False Positive Rate. 

* Choose MODEL_NAME from kFP, Kitsune, Whisper, DF, YaTC and TrafficFormer. The settings of CertTA's smoothing parameters for these models are provided in the `/CertTA_public/evaluation/config/` directory.

* For each flow, a dictionary instance will be saved to record the information required for accuracy meassurement, such as the original flow label and the predicted class. When the predicted class is -1, the flow is detected as an anomaly by the anomaly detector. Otherwise, the predicted class represents the prediction of the CertTA-ceritfied traffic analysis model. These classification results of test flows will be saved as json files in the `/CertTA_public/model/MODEL_NAME/save/DATASET_NAME/` directory. 