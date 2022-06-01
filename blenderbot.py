from typing import List, Text

from transformers import (
    pipeline,
    Conversation,
    BlenderbotTokenizer as Tokenizer, 
    BlenderbotForConditionalGeneration as Blenderbot
)

BLENDERBOT_CONTEXT_LENGTH = 128


class Talker:
    def __init__(self, device=-1) -> None:
        self.device = device
        self.__setup_model()
    
    def __setup_model(self):
        name = "facebook/blenderbot-400M-distill"

        self.model: Blenderbot = Blenderbot.from_pretrained(name)
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(name)

        self.pipeline = pipeline(
            task="conversational", 
            device=self.device,
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

            max_length=BLENDERBOT_CONTEXT_LENGTH,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response: Text = conversation.generated_responses[-1]

        return self.__remove_extra_spaces(response)
    
    @staticmethod
    def __remove_extra_spaces(s: Text) -> Text:
        return ' '.join(s.split())