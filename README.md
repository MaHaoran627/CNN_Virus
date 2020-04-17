# A multi-task CNN model for taxonomic assignment of human viruses

## About data

- The pre-trained model can be found under data folder
- Benchmarking files are also avaliable under data folder
- 4 real RNA-Seq datasets from ncbi are avaliable under data/real-world-data folder
- Training files are not added under data folder, but are avaliable via Google Drive (https://drive.google.com/open?id=1sj0-NCSKjLta_Geg6EMo26rChmtWcOiI)


## To use the pre-trained model to repeat benchmarking results:

python evaluation_on_simulated_50mers.py # benchmarking on 50-mers

python evaluation_on_simulated_100mers.py # benchmarking on 100-mers

python predict_on_realdata.py # prediction on one of the four real-world datasets, modified the file path to have results for other three datasets

python predict_50mers.py # prediction on files that only contain 50-mer

python predict_100mers.py # prediction on files that only contain 50-mer

It is noticed that predict_on_realdata.py can be conducted for dealing K-mers with any length in .fastq format
