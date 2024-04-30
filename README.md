# ATCS practical2 - bias in multilingual dialogue generation

## Setup

Log in to Huggingsface to get the access for the models.

Save target messages to 'messages.pkl' 


## File Structure  

'main.ipynb' - for people who have local GPU

'models.py' - contains the class for 3 multilingual language models (MLM): LLama, MT5, Mixtral

'run.py' - transferred the code from 'main.ipynb' to .py format to enable the generation in Snellius

'run.job' - execure this file to run 'run.py' with GPU