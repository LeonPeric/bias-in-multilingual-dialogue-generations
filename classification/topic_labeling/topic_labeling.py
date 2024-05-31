import pickle

models = ["LLama", "Mistral", "Aya"]
languages = ["English", "Dutch", "Chinese", "Italian"]

templates_list = []
for language in languages:
    with open(f"../../templates/templates_{language}.pkl", "rb") as f:
        templates = pickle.load(f)
        for template in templates:
            templates_list.append(template[0]["content"])

model_output_list = {"LLama": [], "Mistral": [], "Aya": []}

pre_text = "Is the following text specific or general? The text is: "
for model in models:
    results = []
    for language in languages:
        with open(f"../../out/results_{model}_{language}.pkl", "rb") as f:
            outputs = pickle.load(f)

        for item in outputs:
            model_output_list[model].append(item)
            results.append([{"role": "user", "content": pre_text + item}])

    with open(f"../../templates/general_vs_specific_{model}.pkl", "wb") as f:
        pickle.dump(results, f)


model_classification_list = {"LLama": [], "Mistral": [], "Aya": []}
for model in models:
    with open(f"results_topic_labeling_{model}.pkl", "rb") as f:
        classifications = pickle.load(f)
        for classification in classifications:
            model_classification_list[model].append(classification)

topics = [
    "Care",
    "Sympathy",
    "Patronising",
    "Disbelief",
    "Curiosity",
    "None of the above",
]
specific_texts = [
    "I think I have identified the topic:",
    "I believe the topic is:",
    "I'm going to choose option 6:",
    "I believe it is discussing topic 5:",
    "I'm going to take a guess that the topic is:",
    "I can confidently say that the topic is:",
    "I'm convinced that the topic is:",
    "I'm confident that the topic is:",
    "I would say that the topic is:",
    "I would say that the topic is not",
    "After analyzing the text, I can confidently say that it is not discussing any of the topics you listed",
]

classified = {"LLama": [], "Mistral": [], "Aya": []}
for model in models:
    parsed_counts = 0
    for item in model_classification_list[model]:
        before_count = len(classified[model])
        topic_count = 0
        text_count = -1
        for topic in topics:
            if topic in item:
                topic_count += 1
        if topic_count == 1:
            for topic in topics:
                if topic in item:
                    classified[model].append(topic)

        if topic_count > 1:
            text_count = 0
            for text in specific_texts:
                if text in item:
                    text_after = (
                        item.split(text)[1][:13]
                        .replace("\n", "")
                        .replace("*", "")
                        .replace(" ", "")
                        .replace(":", "")
                    )
                    if "ymp" in text_after:
                        text_after = "Sympathy"
                    elif "Curio" in text_after:
                        text_after = "Curiosity"
                    elif "Care" in text_after:
                        text_after = "Care"
                    elif "None" in text_after:
                        text_after = "None of the above"
                    elif "not" in text_after:
                        text_after = "None of the above"
                    elif text_after == ".":
                        text_after = "None of the above"
                    elif text_after == ".Thetextis":
                        text_after = "None of the above"
                    elif "ofthe" in text_after:
                        text_after = "None of the above"
                    elif "about" in text_after:
                        text_after = "None of the above"
                    elif "relatedtoa" in text_after:
                        text_after = "None of the above"
                    classified[model].append(text_after)
                    text_count += 1
            if text_count > 0:
                parsed_counts += 1
        else:
            parsed_counts += 1
        if before_count == len(classified[model]):
            classified[model].append("None of the above")

for model in models:
    with open(f"classification_{model}.csv", "w") as f:
        f.write("id,model_id,template_id,template,output,classification\n")

        for i in range(len(templates_list)):
            model_output = (
                model_output_list[model][i].replace("\n", "\\n").replace('"', "'")
            )
            f.write(
                f'{i},"{model}",{i},"{templates_list[i]}","{ model_output}","{classified[model][i]}"\n'
            )
