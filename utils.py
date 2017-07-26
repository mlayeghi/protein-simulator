import numpy as np

def load_data(orig_seqs, seq_length):
    '''
    Load & prepare fasta sequences.
    '''
    orig_seqs = " ".join(orig_seqs)
    chars = list(set(orig_seqs))
    vocab_size = len(chars)

    # Index to character & character to index dicts for encoding
    # & decoding text into digits: word embedding
    # We need to embed characters as numerical values for the algorithm.
    # We need decoding to convert predicted values back to characters.
    vocab_embed = dict(zip(chars, range(vocab_size)))
    vocab_decode = dict(zip(range(vocab_size), chars))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    # Input & target sequences with required shape for keras input
    # Target sequence is bascally same as input, but shifted by the
    # length of a single character. Thus, the 1st char in target is
    # what naturally follows the 1st char in input, and so forth. So,
    # we used this approach to train our predictive model.
    seqs_x = np.zeros((len(orig_seqs)//seq_length, seq_length, vocab_size))
    seqs_y = np.zeros((len(orig_seqs)//seq_length, seq_length, vocab_size))

    # Basically, breaking the seqs into the desired seq length for
    # training & then generating new seqs of the same length.
    # It's like, we slide a window the size of the desired protein length
    # on seqs, extract the amino acids & apply embedding.
    for i in range(len(orig_seqs)//seq_length):
        # Word embedding; one-hot encoding
        embed_x = [vocab_embed[v] for v in orig_seqs[i*seq_length:(i+1)*seq_length]]
        seqs_x[i, :, :] = np.array([vocab_one_hot[j, :] for j in embed_x])

        embed_y = [vocab_embed[v] for v in orig_seqs[i*seq_length+1:(i+1)*seq_length+1]]
        seqs_y[i, :, :] = np.array([vocab_one_hot[j, :] for j in embed_y])

    return seqs_x, seqs_y, vocab_size, vocab_decode

def read_fasta(infasta):
    '''
    Reading a fasta file one sequence at a time.
    '''
    name, seq = None, []
    for line in infasta:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield(name.replace(">", ""), ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield(name.replace(">", ""), ''.join(seq))

def generate_seq(rnn_model, seq_len, vocab_size, vocab_decode):
    '''
    The function to stitch together the predicted characters and decode them,
    to generate protein sequences of the desired length.
    '''
    # start with a random predicted residue
    pred = np.random.randint(vocab_size)
    # Initialize the array for predicted amino acids.
    # Decode random value to protein residue.
    pr_seq = [vocab_decode[pred]]
    # Create embedding for prediction; needed for algorithm to work
    chars = np.zeros((1, seq_len, vocab_size))
    for i in range(seq_len):
        # Update the embedding
        chars[0, i, :][pred] = 1
        # Predict: one residue at a time
        pred = np.argmax(rnn_model.predict(chars[:, :i+1, :])[0], 1)[-1]
        # Append latest predicted amino acid to the array.
        # Decoding predicted numerical value back to amino acid code.
        pr_seq.append(vocab_decode[pred])
    # Convert the array of amino acids to a protein sequence
    return ('').join(pr_seq)

def write_fasta(proteins, output_file):
    '''
    Function to write generated protein sequences into a fasta file.
    '''
    with open(output_file, 'w') as outfile:
        for i, seq in enumerate(proteins):
            outfile.write(">Seq_{}\n{}\n".format(i+1, seq))

        print("\n\n\t===============================================\n")
        print("\tGenerated proteins are saved in the fasta file:\n\n\t{}".
              format(output_file))

        print("\n\t===============================================\n")
