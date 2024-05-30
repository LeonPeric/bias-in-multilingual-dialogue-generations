# ATCS practical2 - bias in multilingual dialogue generation

This project investigate into the bias in multilingual dialogue generation with 3 multilingual models: Llama, MT5, Mixtral and 4 target languages: English, Dutch, Italian, Chinese.
The 'qr-code.png' contains the QR code linked to the poster of this project.

## Setup

Log in to Huggingsface to get the access for the models.

Enter the following command to install the python environment:

```
conda env create -f dl2023_gpu.yml
```



## File Structure  


'models.py' - contains the class for 3 multilingual language models (MLM): LLama, MT5, Mixtral

'run.py' - transferred the code from 'main.ipynb' to .py format to enable the generation in Snellius

'main.ipynb' - contains the code to call and run the models, it is suitable if the local machine has GPU. Otherwise, use 'main.job' file to generate the dialogues.

'main.job' - execure this file to run 'run.py' with GPU

## Data & analysis

### 'template' folder 

This folder stores the templates, perspective analysis and the generated results.

'descriptor.csv' - contains the descriptor dataset which is translated into 4 languages in this study.

'Processed_Prompts_Translations.csv' - cantains all of the potential combination of templates and the descriptors in 4 languages to generate the dialogue.

'template.ipynb' - contains the perspective and LLM classification analysis result.

The 'pkl' files stores the results provided by the code in the 'template.ipynb' file.

### 'perspective' folder

This folder contains toxicity analysis, where the code can be found in 'main.ipynb'

The analysing resultes can be found in 'toxicity_[language].csv' file and the corresponding plots are stored in 'plots' folder.

### 'out' folder

This folder stores the output of 3 LLM models for 4 languges. The results are stored in the files with the naming format: 'results_[Model Name]_[Language].pkl'.

### 'annotations' folder

This folder stores the annotated output from Llama model in the formate of: "id, model_id, template, output", as a preparation for the perspective analysis.

### 'LLM_Classification' folder

This folder contains the analtysing results for general and specific 
