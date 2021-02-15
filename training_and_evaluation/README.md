# Scripts for training and evaluating the age prediction model

## Requirements

This code was tested on Python 3 with Pytorch 1.2. It uses `numpy`, `pandas`, 
`h5py` for  loading and processing the data and `matplotlib` and `seaborn`
for the plots. See `requirements.txt` to see a full list of requirements
and library versions.


## Folder content


- ``train.py``: Script for training the neural network. To train the neural network run:
```bash
$ python train.py PATH_TO_HDF5 PATH_TO_CSV
```


- ``evaluate.py``: Script for generating the neural network predictions on a given dataset.
```bash
$ python predict.py PATH_TO_MODEL PATH_TO_HDF5_ECG_TRACINGS PATH_TO_CSV  --ouput PATH_TO_OUTPUT_FILE 
```


- ``resnet.py``: Auxiliary module that defines the architecture of the deep neural network.


- ``formulate_problem.py``: Script that separate patients into training, validation and 
```bash
$ python predict.py PATH_TO_CSV 
```

OBS: Some scripts depend on the `resnet.py` and `dataloader.py` modules. So we recomend
the user to, either, run the scripts from within this folder or add it to your python path.
