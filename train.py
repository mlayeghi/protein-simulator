import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
import utils

def main():
    '''
    Main function: set the parameters & call training.
    Training parameters can be adjusted here.
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file', type=str, default='data/protein_samples.fa',
                        help='The address to the input fasta file.')
    parser.add_argument('-output_file', type=str, default='data/generated_proteins.fa',
                        help='The address to write the generated fasta sequences.')
    parser.add_argument('-batch_size', type=int, default=50,
                        help='Minibatch size.')
    parser.add_argument('-num_layers', type=int, default=3,
                        help='Number of layers in the RNN.')
    parser.add_argument('-hidden_dim', type=int, default=100,
                        help='Size of RNN hidden state.')
    parser.add_argument('-seq_length', type=int, default=100,
                        help='RNN sequence length.')
    parser.add_argument('-train_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('-generate_epochs', type=int, default=50,
                        help='Number of generating epochs.')

    args = parser.parse_args()
    train(args)

def train(args):
    '''
    This is where the model is created & trained.
    '''
    # Loading the protein fasta sequences
    orig_seqs = []
    with open(args.input_file, 'r') as infasta:
        for _, seq in utils.read_fasta(infasta):
            orig_seqs.append(seq)
    
    # Prepare training data
    data_x, data_y, vocab_size, vocab_decode = \
        utils.load_data(orig_seqs, args.seq_length)

    # Creating the RNN-LSTM model
    rnn_model = Sequential()
    # Add first layer with input shape provided
    rnn_model.add(LSTM(args.hidden_dim,
                       input_shape=(None, vocab_size),
                       return_sequences=True))
    # Add the remaining layers
    for i in range(args.num_layers - 1):
        rnn_model.add(LSTM(args.hidden_dim,
                           return_sequences=True))
    # We treat each char in the vocbulary as an independet time step
    # and use TimeDistributed wrapper to apply the same dense layer (with
    # number of units = vocab_size) and same weights to output one
    # time step from the sequence for each time step in the input.
    # In other words, we will process all the time steps of the input
    # sequence at the same time.
    rnn_model.add(TimeDistributed(Dense(vocab_size)))
    rnn_model.add(Activation('softmax'))
    rnn_model.compile(loss="categorical_crossentropy",
                      optimizer="rmsprop")

    # Train the model
    rnn_model.fit(data_x,
                  data_y,
                  batch_size=args.batch_size,
                  verbose=1,
                  epochs=args.train_epochs)

    # Generate new sequences
    proteins = []
    for i in range(args.generate_epochs):
        new_pep = utils.generate_seq(rnn_model,
                                     args.seq_length,
                                     vocab_size,
                                     vocab_decode)
        proteins.append(new_pep)

    # Write new protein sequences to a fasta file
    utils.write_fasta(proteins, args.output_file)

if __name__ == '__main__':
    main()
