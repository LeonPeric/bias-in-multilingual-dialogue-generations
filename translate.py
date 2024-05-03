import translators as ts
import json
import os
import time

path = r"ResponsibleNLP\holistic_bias\dataset" # Path to holistic bias dataset
translator = 'google'
version = 'v1.1'
target_lang = 'nl'

target_path = os.path.join(path, version + '-' + target_lang)
if not os.path.exists(target_path):
    os.makedirs(target_path)

def Translate(word, translator, target_lang, n_attempts=10, sleep_time=1): # Uses translators module to translate word, retrying a specified number of times in the case of a connection error
    for attempt in range(n_attempts):
        try:
            out = ts.translate_text(word, translator=translator, from_language='en', to_language=target_lang)
        except:
            print("Connection error on attempt", str(attempt) +",", "retrying in", sleep_time, "second(s)")
            time.sleep(sleep_time)
            continue
        break

    return out

if False:
    with open(os.path.join(path,version,'nouns.json'), 'r', encoding='utf-8') as file:
        nouns = json.load(file)
    for cat in nouns:
        for i in range(len(nouns[cat])): # List of lists containing singular and plural versions of nouns
            sg = Translate(nouns[cat][i][0], translator, target_lang)

            pl = Translate(nouns[cat][i][1], translator, target_lang)
            nouns[cat][i] = [sg, pl]

    with open(os.path.join(target_path,'nouns.json'), 'w', encoding='utf-8') as file:
        json.dump(nouns, file, indent=4)

if False:
    with open(os.path.join(path,version,'descriptors.json'), 'r', encoding='utf-8') as file:
        descriptors = json.load(file)
        for cat in descriptors:
            for subcat in descriptors[cat]:
                for i in range(len(descriptors[cat][subcat])): # List of words, but might also be dictionaries
                    word = descriptors[cat][subcat][i]
                    if isinstance(word, dict):
                        if list(word.keys())[0] == 'descriptor':
                            descriptors[cat][subcat][i]['descriptor'] = Translate(word['descriptor'], translator, target_lang)
                    else:
                        descriptors[cat][subcat][i] = Translate(word, translator, target_lang)

    with open(os.path.join(target_path,'descriptors.json'), 'w', encoding='utf-8') as file:
        json.dump(descriptors, file, indent=4)

if False: # Gives error if two sentences are translated to the same
    with open(os.path.join(path,version,'sentence_templates.json'), 'r', encoding='utf-8') as file:
        sentence_templates = json.load(file) # Contains keys as sentence templates
        for template in sentence_templates:
            new_template = Translate(template, translator, target_lang)
            sentence_templates[new_template] = sentence_templates[template]
            del sentence_templates[template]

    with open(os.path.join(target_path,'sentence_templates.json'), 'w', encoding='utf-8') as file:
            json.dump(sentence_templates, file, indent=4)

def Placeholder(input, direction): # Replaces program's tags with placeholders or reverts them
    if direction == 'placeholder':
        input = input.replace('{article}', '🙂')
        input = input.replace('{noun}', '🤩')
    if direction == 'original':
        input = input.replace('🙂', '{article}')
        input = input.replace('🤩', '{noun}')
    return input

if True:
    with open(os.path.join(path,version,'standalone_noun_phrases.json'), 'r', encoding='utf-8') as file:
        nps = json.load(file)
        for cat in nps:
            for i in range(len(nps[cat])): # List of nps, but might also be dictionaries
                np = nps[cat][i]
                if isinstance(np, dict):
                    sg = Placeholder(np['noun_phrase'], 'placeholder')
                    nps[cat][i]['noun_phrase'] = Placeholder(Translate(sg, translator, target_lang), 'original')
                    if 'plural_noun_phrase' in np:
                        pl = Placeholder(np['plural_noun_phrase'], 'placeholder')
                        nps[cat][i]['plural_noun_phrase'] = Placeholder(Translate(pl, translator, target_lang), 'original')
                else:
                    nps[cat][i] = Placeholder(Translate(Placeholder(np, 'placeholder'), translator, target_lang), 'original')

    with open(os.path.join(target_path,'stand_alone_noun_phrases.json'), 'w', encoding='utf-8') as file:
        json.dump(nps, file, indent=4)