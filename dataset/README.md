# Datasets

There are two directories for each dataset (CICDOH20 and TIISSRC23):

1. `json/`

  *  `statistics.json` contains the total number of flows in the training/test/validation set, the number of classes and the number of flows for each individual class.
  *  `train.json`, `test.json` and `valid.json` contain the flow records of the training/test/validation set, respectively. Each flow record is a dictionary instance containing metadata such as the directional packet length sequence, timestamp sequence, number of packets in the flow, flow label and PCAP file path.

2. `pcap/`

  *  Each class in the dataset corresponds to a subdirectory in `pcap/`, which contains the PCAP files for 5-tuple-identified flow sampels.
  *  Due to the repository size limit, these PCAP files are not provided in our [Github repository](https://github.com/InspiringGroup-Lab/CertTA). The complete artifiacts can be accessed in our [Zenodo repository](https://zenodo.org/records/15580293?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc0ZTYzZmUwLTVlYjQtNDZmOS1iNzM2LTkzZmRmMTAzM2ZlOCIsImRhdGEiOnt9LCJyYW5kb20iOiI5ZTNhYWRlYjBhOTI1ODc2ZDdlNDZlNmM5NDhiZTY4NiJ9.4JTcwwJq-2y3GQIsXA4sEMCbY98XN_HqBM6ws93WXXG3fsCWLH9OlVID2bK8w9RjzRG0kWZrFJmyeU5NGO8lVA).
