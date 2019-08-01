from textgenrnn import textgenrnn
from config import *
from datetime import datetime

model_path = OUTPUT_FOLDER + model_name

textgen = textgenrnn(weights_path=model_path + '_weights.hdf5',
                     vocab_path=model_path + '_vocab.json',
                     config_path=model_path + '_config.json')

temperature = [1.0, 0.5, 0.2, 0.2]
prefix = None  # if you want each generated text to start with a given seed text

if train_cfg['line_delimited']:
    n = 1000
    max_gen_length = 60 if model_cfg['word_level'] else 300
else:
    n = 1
    max_gen_length = 500 if model_cfg['word_level'] else 100

timestring = datetime.now().strftime('%Y%m%d_%H%M%S')
gen_file = '{}_gentext_{}.txt'.format(model_name, timestring)

textgen.generate_to_file("./outputs/" + gen_file,
                         temperature=temperature,
                         prefix=prefix,
                         n=n,
                         max_gen_length=max_gen_length)
