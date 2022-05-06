# clinicaldata-fusion
Documentation
Citation to the original paper  .................................done
Link to the original paper’s repo (if applicable) ................ done
Dependencies --------------------------software requirement
Data download instruction ------------------ done
Preprocessing code + command (if applicable)--------- code?? done
Training code + command (if applicable)
Evaluation code + command (if applicable)
Pretrained model (if applicable)
Table of results (no need to include additional experiments, but main reproducibility result should be included)


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
        - **demo.csv**: 
        - **earlynotes.csv**:
        - **features.csv**:
        - **labels_icd.csv**:
        - **llos.csv**:
        - **los.csv**:
        - **mortality.csv**:
        - **readmit.csv**:
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

| Model    |    Data   |   F1   |  AUROC |  AUPRC |  CPU RAM, Time  |  GPU RAM, Time  |
|----------|:---------:|-------:|-------:|-------:|----------------:|----------------:|
| LR       |  T + S    | 0.6684 | 0.7399 | 0.7303 | 2.40GB, 10.735s | 2.40GB, 10.735s |
| LR       |    U      |        |
| LR       | U + T + S |        |
    
Welcome to contact me <manasag3@illinois.edu> for any question. Partial credit to https://github.com/onlyzdd/clinical-fusion.
