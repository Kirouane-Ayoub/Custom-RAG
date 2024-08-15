import settings
from models import llm_model, llm_tokenizer
from transformers import TextStreamer


class CustomTextStreamer(TextStreamer):
    """
    A custom streamer that prints the output tokens as they are generated.
    """

    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        super().__init__(
            tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens
        )

    def __call__(self, output_ids):
        # Decode and print the output tokens as they are generated
        text = self.tokenizer.decode(
            output_ids, skip_special_tokens=self.skip_special_tokens
        )
        print(text, end="", flush=True)


def pipe(messages: list[dict[str, str]]):
    """
    Processes messages using a tokenizer and model to generate text.

    Parameters:
    - messages (list of str): List of messages to process.
    - tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding and decoding.
    - model (transformers.PreTrainedModel): Model for generating text.
    - device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
    - str: Generated text from the model.
    """
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(settings.DEVICE)
    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    streamer = TextStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        streamer=streamer,
    )

    return generated_ids
