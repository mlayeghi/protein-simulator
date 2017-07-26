##**This code belongs to the "Simulating protein sequences using Recurrent Neural Networks" [post at BioLearning blog.](https://mlayeghi.wordpress.com/2017/07/26/simulating-protein-sequence-using-recurrent-neural-networks/)**

A character-level Long Short-Term Memory (LSTM) RNN model to simulate protein sequences. It takes a multi-sequence fasta file of a set of proteins as input for training and generates a desired number of simulated protein sequences and saves them into a multi-sequence fasta file. User can tune the hyperparameters of the model, for example the number of training epochs before starting to simulate the sequences, in the train.py file.

## Requirements

- Python 3
- Tensorflow > 1.0.1
- keras 2.0.6

## Training

```
python train.py
```
