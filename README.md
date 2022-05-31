## Rasa talkers

Integrates neural network chitchat model into Rasa project. Uses [blenderbot-400m-distill](https://huggingface.co/facebook/blenderbot-400M-distill) (~700mb) for example.

### Description

Rasa project inside `example` folder. It contains default project generated using `rasa init` command. I did few changes:

1. Overrided `action_default_fallback` in `actions/actions.py`
    
    see rasa [documentation](https://rasa.com/docs/rasa/next/fallback-handoff/#3-customizing-the-default-action-optional) for reference

2. Added `action_default_fallback` in `domain.yml`

    ```
    actions:
    - action_default_fallback
    ```

Every time core fallback happens `action_default_fallback` calls Blenderbot model to generate response using latest messages that makes fallback happen.

### How to try

Install dependencies using poetry

```
poetry install
```

Run action server

```bash
# inside repo root
rasa run actions --actions example.actions
```

Run model

```
cd example
rasa train
rasa shell
```