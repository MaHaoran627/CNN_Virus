# A multi-task CNN model for taxonomic assignment of human viruses

In this project, a CNN-based multi-task learning model was developed. This model takes raw K-mers with original sequence information as inputs, and provide two values as outputs------the likelihood values of prediction for taxonomic assignment and for location. A taxonomic report can be generated based on input files by using this model.

## Dependencies
- Keras
- scikit-learn
- GPUs (70GB for training; or decrease the batch_size if memory is not enough)

## About data

- The pre-trained model can be found under data folder
- Benchmarking files are also avaliable under data folder
- 4 real RNA-Seq datasets from ncbi are avaliable under data/real-world-data folder
- Training files are not added under data folder, but are avaliable via Google Drive (https://drive.google.com/open?id=1sj0-NCSKjLta_Geg6EMo26rChmtWcOiI)


## To use the pre-trained model to repeat benchmarking results:

python evaluation_on_simulated_50mers.py # benchmarking on 50-mers

python evaluation_on_simulated_100mers.py # benchmarking on 100-mers

python predict_on_realdata.py # prediction on one of the four real-world datasets, modified the file path to have results for other three datasets

## To use the pre-trained model to do prediction on your own datasets

python predict_50mers.py # prediction on files that only contain 50-mer

python predict_100mers.py # prediction on files that only contain 50-mer

python predict_on_realdata.py # prediction on files contain any length of K-mers. The file must be .fastq format. The speed of prediction will be slower compared with the other two scripts for prediction
