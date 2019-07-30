model_cfg = {
    'word_level': True,  # set to True if want to train a word-level model (requires more data and smaller max_length)
    'rnn_size': 128,  # number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 4,  # number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': True,  # consider text both forwards and backward, can give a training boost
    'max_length': 8,
    # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 100000,  # maximum number of words to model; the rest will be ignored (word-level model only)
    "weights_path": "./outputs/"
}

train_cfg = {
    'line_delimited': True,  # set to True if each text has its own line in the source file
    'num_epochs': 100,  # set higher to train the model for longer
    'gen_epochs': 1,  # generates sample text from model after given number of epochs
    'train_size': 0.9,  # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.0,  # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': True,  # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False,  # set to True if file is a CSV exported from Excel/BigQuery/pandas
    "weights_path": "./outputs/"
}

file_name = "marathon-terminals-v1.txt"
model_name = "marathon_terminal"
