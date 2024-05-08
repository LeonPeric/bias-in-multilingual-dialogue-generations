from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)


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

        if self.model_name == "LLama":
            self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map="auto", use_auth_token=True
            )

        if self.model_name == "Mistral":
            self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map="auto", use_auth_token=True
            )

        if self.model_name == "Aya":
            self.model_id = "CohereForAI/aya-101"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, device_map="auto", use_auth_token=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left", use_auth_token=True
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_input(self, messages) -> list:
        """
        Function for tokenizing batched input, expects the following format:

        """
        input_ids = []

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to("cuda")

        return input_ids

    def generate(self, input_ids) -> list:
        """
        Generate the outputs for a certain input_ids
        """
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # this has to be false when temperature is 0 right?
            # temperature=self.temperature,
            num_return_sequences=self.sequences_amount,
        )

        results = []
        for item in outputs:
            results.append(
                self.tokenizer.decode(
                    item[input_ids.shape[-1] :],
                    skip_special_tokens=True,
                )
            )

        return results
