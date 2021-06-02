# A multi-task CNN model for taxonomic assignment of human viruses

## Update
We have added SARS-CoV-2 genomes from ICTV in the training data and have updated the model. The re-trained model "pretrained_model_add_SARS-CoV-2.h5" has been uploaded in the google drive.
---

In this project, a CNN-based multi-task learning model was developed. This model takes raw K-mers with original sequence information as inputs, and provide two values as outputs------the likelihood values of prediction for taxonomic assignment and for location. A taxonomic report can be generated based on input files by using this model.

## Dependencies
- Keras 2.2.4 under Python 3.7.3
- scikit-learn
- GPUs (70GB for training; or decrease the batch_size if memory is not enough)

## About data
The following data are avaliable via Google Drive (https://drive.google.com/open?id=1sj0-NCSKjLta_Geg6EMo26rChmtWcOiI):

- The pre-trained model
- The mapping list between species names and thier mapped ID
- The weights of species for adjusting the loss in training
- 50-mers and 150-mers simulated from ICTV dataset
- Three simualetd HIV-1 datasets with different divergence
- Four real RNA-Seq datasets from ncbi are avaliable under data/real-world-data folder


## To use the pre-trained model to repeat benchmarking results:

python evaluation_on_simulated_50mers.py # benchmarking on 50-mers

python evaluation_on_simulated_150mers.py # benchmarking on 150-mers

python evaluation_on_simulated_HIV.py # benchmarking on HIV-1 reads

python predict_on_realdata.py # prediction on one of the four real-world datasets, modified the file path to have results for other three datasets
