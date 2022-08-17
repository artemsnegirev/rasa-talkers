from typing import Dict, List, Text

from transformers import (
    pipeline,
    Conversation,
    BlenderbotTokenizer as Tokenizer, 
    BlenderbotForConditionalGeneration as Blenderbot
)

class Talker:
    def __init__(self, device=-1, generate_kwargs: Dict = {}) -> None:
        self.device = device
        self.generate_kwargs = generate_kwargs

        self.__setup_model()
    
    def __setup_model(self):
        # this is the name of checkpoint from huggingface hub
        # https://huggingface.co/facebook/blenderbot-400M-distill

        name = "facebook/blenderbot-400M-distill"

        # replace building conversation input with custom function
        # https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/models/blenderbot/tokenization_blenderbot.py#L77
        Tokenizer._build_conversation_input_ids = tokenize_conversation

        self.model: Blenderbot = Blenderbot.from_pretrained(name)
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(name)

        self.pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="conversational", 
            device=self.device,
            framework='pt'
        )

    def __call__(self, dialog: List[Text]) -> Text:
        conversation = Conversation()

        for idx, utter in enumerate(dialog):
            if idx % 2 == 0:
                conversation.add_user_input(utter)

                if idx != len(dialog) - 1:
                    conversation.mark_processed()
            else:
                conversation.append_response(utter)
        
        conversation: Conversation = self.pipeline(
            conversation,

            max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.eos_token_id,

            **self.generate_kwargs
        )

        response: Text = conversation.generated_responses[-1]

        return self.__remove_extra_spaces(response)
    
    @staticmethod
    def __remove_extra_spaces(s: Text) -> Text:
        return ' '.join(s.split())

def tokenize_conversation(self: Tokenizer, conversation: Conversation) -> List[int]:
    """convert conversation into token_ids and fit to model_max_length"""
    
    full_string = "    ".join([
        text for _, text in conversation.iter_texts()
    ])

    input_ids = self.encode(full_string)

    if len(input_ids) <= self.model_max_length:
        return input_ids

    idx = self.model_max_length
    truncation_idx = None
    last_was_separator = False
    
    while idx > 0 and truncation_idx is None: 
        # the magic number 228 is nothing more than just token_id of 4 spaces
        # Blenderbot uses this token to separate conversation utterances like
        # utter_1    utter_2    utter_3    ...
        # https://github.com/huggingface/transformers/issues/9365#issuecomment-787444332

        if input_ids[-idx] == 228:
            last_was_separator = True
        
        elif last_was_separator:
            truncation_idx = idx

        idx = idx - 1

    if truncation_idx is None:
        truncation_idx = self.model_max_length

    input_ids = input_ids[-truncation_idx:]

    return input_ids