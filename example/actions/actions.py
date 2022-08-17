from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher

from rasa.shared.core.constants import ACTION_DEFAULT_FALLBACK_NAME

from actions.blenderbot import Talker


def get_last_messages(events: List[Dict]) -> List[Text]:
    """gets conversations til user message before latest fallback

    example:
        usr: hello
        bot: Hey! How are you?
        ---- (break)
        usr: Let's talk about you!
        > fallback
        bot: Sure, what would you like to know about me? I'm a pretty interesting person.
        usr: What is your job?
        > fallback
        bot: I work as an accountant. What about you? What do you do for a living?
        usr: I do machine learning magic
    """

    messages = [] 

    start_idx = None 
    fallback_flag = True

    for idx in range(len(events) - 1, -1, -1):
        item = events[idx]

        if (
            item['event'] == 'action' and
            item['name'] == ACTION_DEFAULT_FALLBACK_NAME
        ):
            fallback_flag = True

        if item['event'] == 'user':
            if fallback_flag:
                start_idx = idx
                fallback_flag = False
            else:
                break

    for item in events[start_idx:]:
        if item['event'] in ['user', 'bot']:
            messages.append(item["text"])

    return messages

class ActionBlenderbotTalker(Action):
    def __init__(self) -> None:
        super().__init__()

        self.talker = Talker(
            # optional generation options
            generate_kwargs={
                'num_beams': 10,
                'min_length': 20,
                'no_repeat_ngram_size': 3,
            }
        )

    def name(self) -> Text:
        return ACTION_DEFAULT_FALLBACK_NAME

    def run(
        self, 
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
        ) -> List[Dict[Text, Any]]:

        # collect messages between the bot and the user to pass to the model
        context: List[Text] = get_last_messages(tracker.events)
        # create a new message based on previous utterances and generate options
        new_bot_message: Text = self.talker(context)
        # and finally send message to the user to continue conversation 
        dispatcher.utter_message(new_bot_message)

        return [UserUtteranceReverted()]