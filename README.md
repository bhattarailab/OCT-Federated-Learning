# Investigation of Federated Learning Algorithms for Retinal Optical Coherence Tomography Image Classification with Statistical Heterogeneity

Accepted at IPCAI 2024  
Arxiv Link: [Link](https://arxiv.org/abs/2402.10035)

### Installation 
To begin, set up a Python environment, preferably version 3.10.12, venv. This approach aids in managing dependencies and maintaining a consistent runtime environment.

```
python -m venv env_name
source env_name/bin/activate
```

Once you have created the envrionment, install the required packages from `requirements.txt` using pip:
```
pip install -r requirements.txt
```

### Usage 
Download the OCT dataset from [kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) and execute the following command to run the experiment:
```
python3 train.py --config config_file_path
```
