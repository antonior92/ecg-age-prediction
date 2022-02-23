# Predicting age from the electrocardiogram and its usage as a mortality predictor

Scripts and modules for training and testing deep neural networks for ECG automatic classification.
Companion code to the paper "Deep neural network-estimated electrocardiographic age as a mortality predictor".
https://www.nature.com/articles/s41467-021-25351-7.

Citation:
```
Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a 
mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7. 
```

Bibtex:
```bibtex
@article{lima_deep_2021,
  title = {Deep Neural Network Estimated Electrocardiographic-Age as a Mortality Predictor},
  author = {Lima, Emilly M. and Ribeiro, Ant{\^o}nio H. and Paix{\~a}o, Gabriela MM and Ribeiro, Manoel Horta and Filho, Marcelo M. Pinto and Gomes, Paulo R. and Oliveira, Derick M. and Sabino, Ester C. and Duncan, Bruce B. and Giatti, Luana and Barreto, Sandhi M. and Meira, Wagner and Sch{\"o}n, Thomas B. and Ribeiro, Antonio Luiz P.},
  year = {2021},
  journal = {Nature Communications},
  volume = {12},
  doi = {10.1038/s41467-021-25351-7},
  annotation = {medRxiv doi: 10.1101/2021.02.19.21251232},}
}
```
**OBS:** *The three first authors: Emilly M. Lima, Antônio H. Ribeiro, Gabriela M. M. Paixão contributed equally.*



# Data

Three different cohorts are used in the study:

1. The `CODE` study cohort, with n=1,558,415 patients was used for training and testing:
   - exams from 15% of the patients in this cohort were used for testing. This sub-cohort is refered as `CODE-15%`. 
     The `CODE-15\%` dataset is openly available: [doi: 10.5281/zenodo.4916206 ](https://doi.org/10.5281/zenodo.4916206).
   - the remainign 85%  of the patients were used for developing the neural network model. 
     The full CODE dataset that was used for training is available upon 
     request for research purposes: [doi: 10.17044/scilifelab.15169716](https://doi.org/10.17044/scilifelab.15169716)
2. The `SaMi-Trop` cohort, with n=1,631 patients, is used for external validation.
    - The dataset is openly available: [doi: 10.5281/zenodo.4905618](https://doi.org/10.5281/zenodo.4905618)
3. The `ELSA-Brasil` cohort with n=14,236 patients, is also used for external validation.
    - Request to the ELSA-Brasil cohort should be forward to the ELSA-Brasil Steering Committee.

# Training and evaluation

The code training and evaluation is implemented in Python, contains
  the code for training and evaluating the age prediction model.

## Model

The model used in the paper is a residual neural network. The architecture implementation 
in pytorch is available in `resnet.py`. It follows closely 
[this architecture](https://www.nature.com/articles/s41467-020-15432-4), except that there is no sigmoid at the last layer.

![resnet](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-020-15432-4/MediaObjects/41467_2020_15432_Fig3_HTML.png?as=webp)

The model can be trained using the script `train.py`. Alternatively, 
pre-trained weighs trained on the code dataset for the model described in the paper 
is available in [doi.org/10.5281/zenodo.4892365](https://doi.org/10.5281/zenodo.4892365)
in the following dropbox mirror
[here](https://www.dropbox.com/s/thvqwaryeo8uemo/model.zip?dl=0).
Using the command line, the weights can be downloaded using:
```
wget https://www.dropbox.com/s/thvqwaryeo8uemo/model.zip?dl=0 -O model.zip
unzip model.zip
```
- model input: `shape = (N, 12, 4096)`. The input tensor should contain the 4096 points of the ECG tracings sampled at 400Hz (i.e., a signal of approximately 10 seconds). Both in the training and in the test set, when the signal was not long enough, we filled the signal with zeros, so 4096 points were attained. The last dimension of the tensor contains points of the 12 different leads. The leads are ordered in the following order: {DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}. All signal are represented as 32 bits floating point numbers at the scale 1e-4V: so if the signal is in V it should be multiplied by 1000 before feeding it to the neural network model.
- model output: `shape = (N, 1) `. With the entry being the predicted age from the ECG.

## Requirements

This code was tested on Python 3 with Pytorch 1.2. It uses `numpy`, `pandas`, 
`h5py` for  loading and processing the data and `matplotlib` and `seaborn`
for the plots. See `requirements.txt` to see a full list of requirements
and library versions.

**For tensorflow users:** If you are interested in a tensorflow implementation, take a look in the repository:
https://github.com/antonior92/automatic-ecg-diagnosis. There we provide a tensorflow/keras implementation of the same 
resnet-based model. The problem there is the abnormality classification from the ECG, nonetheless simple modifications 
should suffice for dealing with age prediction

## Folder content


- ``train.py``: Script for training the neural network. To train the neural network run:
```bash
$ python train.py PATH_TO_HDF5 PATH_TO_CSV
```


- ``evaluate.py``: Script for generating the neural network predictions on a given dataset.
```bash
$ python evaluate.py PATH_TO_MODEL PATH_TO_HDF5_ECG_TRACINGS --output PATH_TO_OUTPUT_FILE 
```


- ``resnet.py``: Auxiliary module that defines the architecture of the deep neural network.


- ``formulate_problem.py``: Script that separate patients into training, validation and 
```bash
$ python predict.py PATH_TO_CSV 
```

- ``plot_learning_curves.py``: Auxiliary script that plots learning curve of the model.
```bash
$ python plot_learning_curves.py PATH_TO_MODEL/history.csv
```

OBS: Some scripts depend on the `resnet.py` and `dataloader.py` modules. So we recomend
the user to, either, run the scripts from within this folder or add it to your python path.
