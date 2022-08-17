# Towards open-domain conversation in Rasa: Blenderbot example

## Intro
 
There are task-oriented and chitchat (open-domain) bots. [Rasa](https://github.com/RasaHQ/rasa/) is great to develop task-oriented bots. Sometimes it is needed to handle chitchat. There are retrieval based and generative based approaches. Rasa has components to handle single-turn [chitchat](https://rasa.com/docs/rasa/chitchat-faqs) using retrieval approach. Retrieval-based approach limited by amount of human designed queries and responses. Retrieval chitchat usually well-designed, and we can control dialog. In a generative paradigm, you don't need to design queries and responses because models already pretrained to talk, but you can not control what model will talk about. Advantage is generative-based models can respond to any given context. [Blenderbot](https://parl.ai/projects/recipes/) is near human level talker, but is not safe for production out of the box. [Blenderbot 2.7B](https://huggingface.co/facebook/blenderbot-400M-distill) pretrained model checkpoint freely available on huggingface. This is example how to extend Rasa chitchat capabilities with easy. You could try generative-based chitchat if it fits your user experience: it's okay about factual mistakes, harmful stereotypes, and you really need something funny and open-domain.


## Tutorial

Rasa uses policies to predict action, which can be simple utterance or custom code. So we have to 1) write an action that uses `Blenderbot` model and 2) use policies to predict our action.

### Write custom action to use Blenderbot

As `Blenderbot` is a context-aware chitchat model, we can use conversation context to create specific and engaged response.

```python
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher

from rasa.shared.core.constants import ACTION_DEFAULT_FALLBACK_NAME

from blenderbot import Talker

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

    # see implementation details in repo
    ...
```

Also, we need to implement generation function to use `Blenderbot`. Have you noticed a `Talker` class? This is exactly what we need. Internally, it uses `pipeline` from `transformers` library that makes it simple to use. I just added a little customization to prepare conversation with `tokenize_conversation` function.

The most exciting part is the `pipeline` function that allows us to create pipeline which can use a lot of models from huggingface hub like [DialoGPT](https://github.com/microsoft/DialoGPT) and others. You can find all available models [here](https://huggingface.co/models?pipeline_tag=conversational) by filtering them with `conversational` tag. It will require some customization to user others models, but this is a point for improvements!

```python
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

    # see implementation details in repo
    ...
```

### Using action in Rasa Policies

We can use action in different ways:
- as fallback action in RulePolicy
- as response action on chitchat/out_of_scope intent
- as action to predcit using TED policy in e2e fashion

#### Fallback action

You could notice that I used `ACTION_DEFAULT_FALLBACK_NAME` variable to name our custom action. This name reserved for `action_default_fallback`. By default, this action is predicted by the RulePolicy after core fallback. Please refer to [fallback mechanism](https://rasa.com/docs/rasa/fallback-handoff#handling-low-action-confidence). And all I need to do is just to register action in `domain.yml` file:

```yaml
actions:
  - action_default_fallback
```

So now every time core fallback happens, our custom action calls `Blenderbot` model to generate response using latest messages that make fallback happen. It can be very useful when we build a funny task-oriented bot and when fallback happens the bot should continue conversation using context information to answer.

#### Handle specific intent

Another option is to use action as a response to developer defined intent using a rule. The advantage of that option is the developer explicitly controls topics to handle. The downside is contradiction with general intents like affirm/deny that sometimes appear in chitchat conversation too. But we could just write some rules to handle that behavior. For example, write rule that allow to continue chitchat even non-chitchat intent occurs. Another great thing is that you can add standard retrieval Rasa components to handle specific topics with predefined responses, in addition to Blenderbot.

```yaml
# add new rules in data/rules.yml
- rule: Handle chitchat with Blenderbot action
  steps:
  - intent: chitchat
  - action: action_default_fallback

- rule: Continue chitchat on affirm/deny
  steps:
  - action: action_default_fallback
  - or: 
    - intent: affirm
    - intent: deny
  - action: action_default_fallback

# extend intent list in domain.yml
intents:
  ... 
  - chitchat

# and add some examples in data/nlu.yml
- intent: chitchat
  examples: |
    - Can I ask you questions first?
    - Can I get a hamburger?
    - Can you make sandwiches?
    - Can you please send me an uber
    ...
    - I am an opioid addict
    - I am hungry
    - I wan to buy a plane
    - I want french cuisine
```

#### Using contextual prediction with TED

Usually we put each query into well separated classes called `intents`. But there are cases when we can't assign class without context, as listed below.  

```
usr: Hello
bot: Hey! How are you?
usr: I am sad
bot: Here is something to cheer you up:
     Image: https://i.imgur.com/nGF1K8f.jpg
     Did that help you?
usr: I think I saw this before <----- chitchat/deny/?
bot: Really? What did you think of it? I thought it was pretty good, but I didn't expect it to be that good.
```

Rasa provides [e2e](https://rasa.com/blog/were-a-step-closer-to-getting-rid-of-intents/) learning with [TED](https://rasa.com/blog/unpacking-the-ted-policy-in-rasa-open-source/) policy, which uses ML to predict next action. To use e2e learning, I added some examples with raw user input.

```yml
# data/stories.yml

- story: e2e example 1
  steps:
  - intent: greet
  - action: utter_greet
  - intent: mood_unhappy
  - action: utter_cheer_up
  - action: utter_did_that_help
  - user: I saw this picture before
  - action: action_default_fallback

# ... other examples
```

I found [dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) very useful, especially for TED policy to reduce overfitting. Also, you should set high `e2e_confidence_threshold` to avoid inappropriate Blenderbot prediction.

```yml
# config.yml

policies:
  - name: TEDPolicy
    epochs: 25
    drop_rate_label: 0.4
    drop_rate_dialogue: 0.4
    drop_rate_attention: 0.4
    e2e_confidence_threshold: 0.8
    constrain_similarities: true

  # ... other polices
```

### How to try

Install dependencies using poetry

```
poetry install
```

Run action server

```bash
# Terminal 1

cd example
rasa run actions --actions example.actions
```

Run model

```bash
# Terminal 2

cd example
rasa train
rasa shell
```