# clinicaldata-fusion

## Data and Code for the paper "Combining structured and unstructured data for predictive models: a deep learning approach"

This repository contains data and source code for paper *Combining structured and unstructured data for predictive models: a deep learning approach*. In this paper, two frameworks are proposed, namely *Fusion-CNN* and *Fusion-LSTM*. In this method, sequential clinical notes and temporal signals are combined for patient outcome prediction. Three prediction tasks: *In-hospital mortality prediction*, *long length of stay prediction*, and *30-day readmission prediction* on MIMIC-III datasets empirically shows the effectiveness of proposed models. Combining structured and unstructured data leads to a significant performance improvement.

### Framework

![Fusion-CNN](https://imgur.com/nKhAOrM.png)

> Fusion-CNN is based on document embeddings, convolutional layers, max-pooling layers. The final patient representation is the concatenation of the latent representation of sequential clinical notes, temporal signals, and the static information vector. Then the final patient representation is passed to output layers to make predictions.

![Fusion-LSTM](https://imgur.com/AgrIkl6.png)

> Fusion-LSTM is based on document embeddings, LSTM layers, max-pooling layers. The final patient representation is the concatenation of the latent representation of sequential clinical notes, temporal signals, and the static information vector. Then the final patient representation is passed to output layers to make predictions.

### Citation to the original paper

```bibtex
@inproceedings{onlyzdd/clinical-fusion,
    title = {Combining structured and unstructured data for predictive models: a deep learning approach},
    author = {Dongdong Zhang, Changchang Yin, Jucheng Zeng, Xiaohui Yuan & Ping Zhang},
    url = {https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01297-6}, 
    doi = {https://doi.org/10.1186/s12911-020-01297-6},
    article number: {280 (2020)},
    year = {Received: 04 April 2020, Accepted: 19 October 2020, Published: 29 October 2020},
    venue/publisher= {BMC Medical Informatics and Decision Making}
}
```

### Link to original paper repository

https://github.com/onlyzdd/clinical-fusion

### Dependencies

  Required Software Packages versions that are required,
  
  - Python 3.6.10
  - Gensim 3.8.0
  - NLTK: 3.4.5
  - Numpy: 1.14.2
  - Pandas: 0.25.3
  - Scikit-learn: 0.20.1
  - Tqdm: 4.42.1
  - PyTorch: 1.4.0

```python
  pip install NLTK
```

```python
  python -m nltk.downloader stopwords
```

```python
  pip install memory_profiler
```

### Data download instruction

MIMIC-III database is analyzed in this study is available on [PhysioNet](https://mimic.physionet.org/about/mimic) repository. Here are some steps to prepare for the dataset:
- To request access to MIMIC-III, please follow https://mimic.physionet.org/gettingstarted/access/.
- Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset
```python
  wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciii/1.4/
```
- Unzip csv files
- Download postgressql, and follow the steps to build the mimic database from  
- With access to MIMIC-III, to build the MIMIC-III dataset locally using Postgres, follow the instructions at https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres
- Run SQL queries to generate views/tables.
- Run SQL queries from the *query* folder that generates the required csv files for this paper.
- Copy the generated csv files to *data/mimic* folder
- Also copy the Admissions.csv, NOTESEVENT.csv from MIMIC-III version 1.4 to *data/mimic* folder

#### Preprocessing

Run these python scripts to generate necessary preprocessed data and stored in *data/processed* folder
```sh
$ python 00_define_cohort.py # define patient cohort and collect labels
$ python 01_get_signals.py # extract temporal signals (vital signs and laboratory tests)
$ python 02_extract_notes.py --firstday # extract first day clinical notes
$ python 03_merge_ids.py # merge admission IDs
$ python 04_statistics.py # run statistics
$ python 05_preprocess.py # run preprocessing
$ python 06_doc2vec.py --phase train # train doc2vec model
$ python 06_doc2vec.py --phase infer # infer doc2vec vectors
```

### Folder Specification
- ```data/```
    - ```mimic/```
        - **adm_details.csv**: generated csv file from admissions and patient table using adm_details.sql in query folder.
        - **DIAGNOSES_ICD.csv**: the diagnosis file from MIMIC-III raw dataset
        - **NOTEEVENTS.csv**: the notes file from MIMIC-III raw dataset
        - **pivoted_lab.csv**: generated csv file from admissions, icustays, labevents table using pivoted-lab.sql in query folder.
        - **pivoted-vital.csv**: generated csv file from admissions, icustays, chartevents table using pivoted-vital.sql in query folder.
    - ```processed/```
        - ```demo.csv, earlynotes.csv, features.csv, labels_icd.csv, llos.csv, los.csv, mortality.csv, readmit.csv```: These files are generated
        - ```files/```
        - ```initial_data/```
        - ```resample_dir/```

- ```models/```
- ```query/```
  - **adm_details.sql**: This file extracts data(patient demographic information) from admissions and patient tables.
  - **pivoted-lab.sql**: This file extracts data(patient lab events) from lab events, icu stays tables.
  - **pivoted-vital.sql**: This file extracts data(patient vital events) from chart events, icu stays tables.
- ```tools/```
  - ```parse.py```: Parse the input arguments from the python command
- **00_define_cohort.py**: preprocessing file: define patient cohort and collect labels
- **01_get_signals.py**: preprocessing file: extract temporal signals (vital signs and laboratory tests)
- **02_extra_notes.py**: preprocessing file: extract first day clinical notes
- **03_merge_ids.py**: preprocessing file: merge admission IDs
- **04_statistics.py**: preprocessing file: run statistics
- **05_preprocess.py**: preprocessing file: run preprocessing
- **06_doc2vec.py**: train and infer doc2vec model
- **README.md**:
- **baselines.py**: This file process input commands for traditional methods (random-forest, logistic regression).
- **ci.py**:
- **cnn.py**: Fusion-CNN architecture
- **data_loader.py**:
- **function.py**:
- **getram.py**:
- **lstm.py**: Fusion-LSTM architecture
- **main.py**: Main file which process input commands for Fusion-LSTM/Fusion-CNN methods. (Deep learning methods).
- **myloss.py**:
- **p_value.py**:
- **py_op.py**:
- **run_baselines.sh**: Might not need this.
- **utils.py**:


### Step 3: run the code
### Run

#### Baselines

Baselines (i.e., logistic regression, and random forest) are implemented using scikit-learn. To run:

```sh
$ python baselines.py --model [model] --task [task] --inputs [inputs]
```
Here's the argument,

```bibtext
usage: baselines.py [-h] [--task TASK] [--model MODEL] [--inputs INPUTS]

optional arguments:

  -h, --help       show this help message and exit
  
  --task TASK      mortality, readmit, llos (default:mortality)
  
  --model MODEL    all, lr, or rf (default:all)
  
  --inputs INPUTS  3: T + S, 4: U, 7: U + T + S (default=4)
  ```

#### Deep models

Fusion-CNN and Fusion-LSTM are implemented using PyTorch. To run:

```sh
$ python main.py --model [model] --task [task] --inputs [input] # train Fusion-CNN or Fusion-LSTM
$ python main.py --model [model] --task [task] --inputs [input] --phase test --resume # evaluate
```
Here's the argument,
```bibtext
usage: main.py [-h] [--inputs INPUTS] [--data-dir DATA_DIR] [--task S]
               [--last-time last event time] [--time-range TIME_RANGE]
               [--n-code N_CODE] [--n-visit N_VISIT] [--model MODEL]
               [--split-num split num] [--split-nor split normal range]
               [--use-glp use global pooling operation]
               [--use-value use value embedding as input]
               [--use-cat use cat for time and value embedding]
               [--embed-size EMBED SIZE] [--rnn-size rnn SIZE]
               [--hidden-size hidden SIZE] [--num-layers num layers]
               [--phase PHASE] [--batch-size BATCH SIZE]
               [--model-path MODEL_PATH] [--resume S] [--workers N] [--lr LR]
               [--epochs N]
clinical fusion help
optional arguments:
  -h, --help            show this help message and exit
  --inputs INPUTS       selected and preprocessed data directory
  --data-dir DATA_DIR   selected and preprocessed data directory
  --task S              start from checkpoints
  --last-time last event time
                        last time
  --time-range TIME_RANGE
  --n-code N_CODE       at most n codes for same visit
  --n-visit N_VISIT     at most input n visits
  --model MODEL, -m MODEL
                        model
  --split-num split num
                        split num
  --split-nor split normal range
                        split num
  --use-glp use global pooling operation
                        use global pooling operation
  --use-value use value embedding as input
                        use value embedding as input
  --use-cat use cat for time and value embedding
                        use cat or add
  --embed-size EMBED SIZE
                        embed size
  --rnn-size rnn SIZE   rnn size
  --hidden-size hidden SIZE
                        hidden size
  --num-layers num layers
                        num layers
  --phase PHASE         train/test phase
  --batch-size BATCH SIZE, -b BATCH SIZE
                        batch size
  --model-path MODEL_PATH
                        model path
  --resume S            start from checkpoints
  --workers N           number of data loading workers (default: 32)
  --lr LR, --learning-rate LR
                        initial learning rate
  --epochs N            number of total epochs to run
  
```
**1. Long Length Stay Predictive Task Metrics (Binary Classification)**


| Model    |    Data   |   F1   |  AUROC |  AUPRC |  CPU RAM, Time     |  GPU RAM, Time    |
|----------|:---------:|-------:|-------:|-------:|-------------------:|------------------:|
| LR       |  T + S    | 0.6684 | 0.7399 | 0.7303 | 2.40GB, 10.735s    |     0GB, 0s       |
| LR       |    U      | 0.7613 | 0.8399 | 0.8206 | 10.92GB, 105.179s  |     0GB, 0s       |
| LR       | U + T + S | 0.7526 | 0.8302 | 0.8143 | 10.94GB, 112.077s  |     0GB, 0s       |
| RF       |  T + S    | 0.7026 | 0.7746 | 0.7580 | 2.47GB, 504.209s   |     0GB, 0s       | 
| RF       |    U      | 0.7586 | 0.8553 | 0.8503 | 10.70GB, 1650.264s |     0GB, 0s       |
| RF       | U + T + S | 0.7747 | 0.8656 | 0.8605 | 10.72GB, 1701.50s  |     0GB, 0s       |
| LSTM     |  T + S    | 0.7111 | 0.7886 | 0.7744 | 11.64GB, 1812.1s   |  1.37GB, 1690.5s  |
| LSTM     |    U      | 0.8000 | 0.8851 | 0.8900 | 12.59GB, 2931.257s |  1.12GB, 2810.5s  |
| LSTM     | U + T + S | 0.8200 | 0.8996 | 0.9022 | 13.93GB, 1839.707s |  1.46GB, 1729.7s  |
| CNN      |  T + S    | 0.7251 | 0.8019 | 0.7884 | 11.27GB, 2983.568s |  1.33GB, 2923.5s  |
| CNN      |    U      | 0.8240 | 0.9079 | 0.9111 | 13.52GB, 2188.728s |  1.08GB, 2120.7s  |
| CNN      | U + T + S | **0.8371** | **0.9197** | **0.9230** | **13.83GB, 2631.743s** |  **1.43GB, 2574.7s**  |

**2. 30-day Readmission Predictive Task Metrics (Binary Classification)**


| Model    |    Data   |   F1   |  AUROC |  AUPRC  |  CPU RAM, Time     |  GPU RAM, Time    |
|----------|:---------:|-------:|-------:|--------:|-------------------:|------------------:|
| LR       |  T + S    | 0.1422 | 0.6566 | 0.0948  | 2.83GB, 9.637s     |  0GB, 0s          |
| LR       |    U      | 0.1770 | 0.7161 | 0.1491  | 11.44GB, 105.658s  |  0GB, 0s          |
| LR       | U + T + S | 0.1545 | 0.6973 | 0.1112  | 11.47GB, 112.091s  |  0GB, 0s          |
| RF       |  T + S    | 0.1342 | 0.6396 | 0.0875. | 1.86GB, 1559.633s  |  0GB, 0s          |
| RF       |    U      | 0.1182 | 0.6205 | 0.0795  | 10.49GB, 1755.992s |  0GB, 0s          |
| RF       | U + T + S | 0.1211 | 0.6302 | 0.0818  | 10.51GB, 1978.742s |  0GB, 0s          |
| LSTM     |  T + S    | 0.1316 | 0.6900 | 0.1005  | 11.82GB, 1799.94s  |  1.45GB, 1640.9s  |
| LSTM     |    U      | 0.1463 | 0.6875 | 0.1078  | 13.92GB, 2955.894s |  1.12GB, 2845.5s  |
| LSTM     | U + T + S | 0.1465 | 0.7019 | 0.1198  | 14.13GB, 1850.01s  |  1.46GB, 1630.0s  |
| CNN      |  T + S    | 0.1681 | 0.7093 | 0.1124  | 13.54GB, 1844.15s  |  1.30GB, 1791.15s |
| CNN      |    U      | 0.1690 | 0.7288 | 0.1294  | 14.20GB, 2029.463s |  1.32GB, 1971s    |
| CNN      | U + T + S | **0.2092** | **0.7958** | **0.1811**  | **14.61GB, 2289.474s** |  **1.98GB, 2236.5s**  |

**3. In-hospital Mortality Predictive Task Metrics (Binary Classification)**


| Model    |    Data   |   F1   |  AUROC |  AUPRC |   CPU RAM, Time    |  GPU RAM, Time    |
|----------|:---------:|-------:|-------:|-------:|-------------------:|------------------:|
| LR       |  T + S    | 0.3008 | 0.7898 | 0.2879 | 2.96GB, 9.727s     |  0GB, 0s          |
| LR       |    U      | 0.5047 | 0.9407 | 0.6927 | 11.40GB, 104.411s  |  0GB, 0s          |
| LR       | U + T + S | 0.4211 | 0.9036 | 0.4935 | 11.42GB, 113.204s  |  0GB, 0s          |
| RF       |  T + S    | 0.3180 | 0.8224 | 0.7303 | 1.86GB, 520.017s   |  0GB, 0s          |
| RF       |    U      | 0.4490 | 0.8999 | 0.5496 | 10.49GB, 273.870s  |  0GB, 0s          |
| RF       | U + T + S | 0.4581 | 0.8987 | 0.5717 | 10.52GB, 1282.628s |  0GB, 0s          |
| LSTM     |  T + S    | 0.3156 | 0.8304 | 0.3288 | 13.79GB, 3268.05s  |  1.45GB, 3118.0s  |
| LSTM     |    U      | 0.4438 | 0.9107 | 0.5276 | 13,60GB, 2991.641s |  1.12GB, 2880.6s  |
| LSTM     | U + T + S | 0.4543 | 0.9174 | 0.5489 | 13.82GB, 4398.216s |  1.46GB, 4278.216s|
| CNN      |  T + S    | 0.3524 | 0.8590 | 0.3747 | 13.55GB, 1836.736s |  1.33GB, 1.782.1s |
| CNN      |    U      | **0.5596** | **0.9589** | 0.6821 | 13.58GB, 2041.955s |  0.98GB, 2001.8s  |
| CNN      | U + T + S | 0.5289 | 0.9527 | 0.6566 | 13.85GB, 2117.753s |  0.98GB, 2059.5s  |


Welcome to contact me <manasag3@illinois.edu> for any question. Partial credit to https://github.com/onlyzdd/clinical-fusion.
