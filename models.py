from transformers import AutoTokenizer, AutoModelForCausalLM, MT5ForConditionalGeneration
import torch


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

        if self.model_name == "MT5":
            self.model_id = "google/mt5-large"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.model_name == "LLama":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
        if self.model_name == "MT5":
            self.model = MT5ForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def prepare_input(self, messages: list) -> list:
        """
        Function for tokenizing batched input, expects the following format:

        """
        input_ids = []

        # quite a large difference in generation when using chat template
        # also big difference in inference time (chat is 3x faster)
        # however it is easier and more universal across models wrt template design
        # for message in messages:
        #     message_ids = self.tokenizer.apply_chat_template(
        #         message, add_generation_prompt=True, return_tensors="pt", padding=True
        #     )
        #     input_ids.append(message_ids)
        # input_ids = torch.stack(input_ids, dim=0).squeeze().to(self.model.device)
        input_ids = self.tokenizer(messages, padding=True, return_tensors="pt").to(self.model.device)
        

        return input_ids["input_ids"]

    def generate(self, input_ids) -> list:
        """
        Generate the outputs for a certain input_ids
        """
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=self.sequences_amount,
        )

        results = []
        for i in range(self.batch_size):
            query_results = []
            for item in outputs[
                i * self.sequences_amount : (i + 1) * self.sequences_amount
            ]:
                query_results.append(
                    self.tokenizer.decode(
                        #item[input_ids.shape[-1] :], skip_special_tokens=True
                        item, skip_special_tokens=True
                    )
                )

            results.append(query_results)

        return results
