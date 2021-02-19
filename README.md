# Predicting age from the electrocardiogram and its usage as a mortality predictor

Scripts and modules for training and testing deep neural networks for ECG automatic classification.
Companion code to the paper:

- **Deep neural network estimated electrocardiographic-age as a mortality predictor.**
*Emilly M Lima, Antônio H Ribeiro, Gabriela MM Paixão, Manoel Horta Ribeiro, Marcelo M Pinto Filho,
Paulo R Gomes, Derick M Oliveira, Ester C Sabino,  Bruce B Duncan, Luana Giatti, Sandhi M Barreto, 
Wagner Meira Jr, Thomas B Schön, Antonio Luiz P Ribeiro.* Under Review.


# Data

Three diferent cohorts are used in the study:

1. The `CODE` study cohort, with n=1,558,415 patients
   - 15% of the patients in this cohort is used for testing and refered as `CODE-15%`.
   - the remainign 85%  of the patients is used for developing the neural network model
2. The `SaMi-Trop` cohort, with n=1,631 patients, is used for external validation
3. The `ELSA-Brasil` cohort with n=14,236 patients, is also used for external validation.


Data avaibility statement:
```
Upon publication,  some of the cohorts used in the model evaluation will be made available. 
Information about mortality, age, sex, the ECG tracings and the flag indicating whether the 
ECG tracing is normal will be made available with no restriction for the Sami-Trop cohort 
and for the CODE-15% cohorts. The DNN model parameters that give the results presented in this
paper will also be made available without restrictions. This should allow the reader to partially 
reproduce the results presented in the paper. Restrictions apply to additional clinical information 
on these two cohorts, to the full CODE cohort and to the ELSA-Brasil cohort, for which requests 
will be considered on an individual basis by the Telehealth Network of Minas Gerais and by ELSA-Brasil
Steering Committee. Any data use will be restricted to non-commercial research purposes, and the data
will only be made available on the execution of appropriate data use agreements.
```


# Training and evaluation

The code training and evaluation is implemented in Python, contains
  the code for training and evaluating the age prediction model.

## Model

The model can be trained using the script `train.py`. Alternatively, 
pre-trained weighs trained on the code dataset for the model described in the paper 
is available in
[here](https://www.dropbox.com/s/thvqwaryeo8uemo/model.zip?dl=0).
Using the command line, the weights can be downloaded using:
```
wget https://www.dropbox.com/s/thvqwaryeo8uemo/model.zip?dl=0 -O model.zip
unzip model.zip
```
The model is described in the module ``
- input: shape = (N, 12, 4096). The input tensor should contain the 4096 points of the ECG tracings sampled at 400Hz (i.e., a signal of approximately 10 seconds). Both in the training and in the test set, when the signal was not long enough, we filled the signal with zeros, so 4096 points were attained. The last dimension of the tensor contains points of the 12 different leads. The leads are ordered in the following order: {DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}. All signal are represented as 32 bits floating point numbers at the scale 1e-4V: so if the signal is in V it should be multiplied by 1000 before feeding it to the neural network model.

- output: shape = (N, 1). With the entry being the predicted age from the ECG.

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
$ python predict.py PATH_TO_MODEL PATH_TO_HDF5_ECG_TRACINGS --ouput PATH_TO_OUTPUT_FILE 
```


- ``resnet.py``: Auxiliary module that defines the architecture of the deep neural network.


- ``formulate_problem.py``: Script that separate patients into training, validation and 
```bash
$ python predict.py PATH_TO_CSV 
```

OBS: Some scripts depend on the `resnet.py` and `dataloader.py` modules. So we recomend
the user to, either, run the scripts from within this folder or add it to your python path.


# Statistical analysis

Implemented in R, code for the statistical analysis presented in the paper will
be released upon publication: together with the `CODE-15%` and `SaMi-Trop` datasets.
