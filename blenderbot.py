from typing import List, Text, Tuple

from transformers import (
    BlenderbotSmallTokenizer as Tokenizer, 
    BlenderbotSmallForConditionalGeneration as Blenderbot
)


class Talker:
    def __init__(self, device='cpu') -> None:
        self.device = device

        self.__setup_model()
    
    def __setup_model(self):
        name = "facebook/blenderbot_small-90M"

        self.model: Blenderbot = Blenderbot.from_pretrained(name)
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(name)

        self.model.to(self.device)

    def __call__(self, context: List[Text]) -> Tuple[Text, List[Text]]:
        context = ['    '.join(context)]

        inputs = self.tokenizer(
            context,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs.to(self.device)

        output = self.model.generate(**inputs)
        output.to('cpu')

        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]