from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

from tqdm import tqdm


class Model:
    """
    General class for the model.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        sequences_amount: int,
        batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.sequences_amount = sequences_amount
        self.batch_size = batch_size

        token = "SECRET"

        if self.model_name == "LLama":
            self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        if self.model_name == "Mistral":
            self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"

        if self.model_name == "Aya":
            self.model_id = "CohereForAI/aya-101"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left", token=token
        )

        task = "text-generation"

        if self.model_name == "Aya":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, token=token
            )
            task = "text2text-generation"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, token=token
            )
        self.model = pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

        self.terminators = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        self.model.tokenizer.pad_token_id = self.model.model.config.eos_token_id

    def prepare_input(self, messages) -> list:
        """
        Function for tokenizing batched input, expects the following format:

        """

        input_ids = self.model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        return input_ids

    def generate(self, input_ids) -> list:
        """
        Generate the outputs for a certain input_ids
        """
        if self.model_name == "Aya":
            outputs = []
            for output in tqdm(
                self.model(
                    input_ids,
                    batch_size=self.batch_size,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.terminators,
                ),
                total=len(input_ids),
            ):
                outputs.append(self.model.tokenizer.decode(output[0]))

            return outputs

        outputs = []
        for output in tqdm(
            self.model(
                input_ids,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                return_full_text=False,
                eos_token_id=self.terminators,
            ),
            total=len(input_ids),
        ):
            outputs.append(output[0]["generated_text"])

        return outputs
