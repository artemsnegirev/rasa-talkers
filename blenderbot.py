from typing import List, Text

from transformers import (
    BlenderbotTokenizer as Tokenizer, 
    BlenderbotForConditionalGeneration as Blenderbot
)

BLENDERBOT_CONTEXT_LENGTH = 128


class Talker:
    def __init__(self, device='cpu') -> None:
        self.device = device
        self.__setup_model()
    
    def __setup_model(self):
        name = "facebook/blenderbot-400M-distill"

        self.model: Blenderbot = Blenderbot.from_pretrained(name)
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(name)

        self.model.to(self.device)

    def _truncate_to_max_length(self, inputs):
        tokens = inputs['input_ids'][0]

        if len(tokens) <= BLENDERBOT_CONTEXT_LENGTH:
            return inputs

        idx = BLENDERBOT_CONTEXT_LENGTH
        truncation_idx = None
        last_was_separator = False
        
        while idx > 0 and truncation_idx is None: 
            if tokens[-idx] == 228:
                last_was_separator = True
            
            elif last_was_separator:
                truncation_idx = idx

            idx = idx - 1

        if truncation_idx is None:
            truncation_idx = BLENDERBOT_CONTEXT_LENGTH

        inputs['input_ids'] = inputs['input_ids'][..., -truncation_idx:]
        inputs['attention_mask'] = inputs['attention_mask'][..., -truncation_idx:]

        return inputs 

    def __call__(self, context: List[Text]) -> Text:
        context = ['    '.join(context)]

        inputs = self.tokenizer(
            context,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        inputs = self._truncate_to_max_length(inputs)

        output = self.model.generate(**inputs)
        output.to('cpu')

        response = self.tokenizer.batch_decode(
            output, 
            skip_special_tokens=True
        )[0]

        return self.__remove_extra_spaces(response)
    
    @staticmethod
    def __remove_extra_spaces(s: Text) -> Text:
        return ' '.join(s.split())