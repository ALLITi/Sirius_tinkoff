---
license: mit
widget:
- text: "@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела? @@ВТОРОЙ@@"
  example_title: "how r u"
- text: "@@ПЕРВЫЙ@@ что ты делал на выходных? @@ВТОРОЙ@@"
  example_title: "wyd"
language:
- ru
tags:
- conversational
---

This generation model is based on [sberbank-ai/rugpt3medium_based_on_gpt2](https://huggingface.co/sberbank-ai/rugpt3medium_based_on_gpt2). It's trained on large corpus of dialog data and can be used for buildning generative conversational agents

The model was trained with context size 3


On a private validation set we calculated metrics introduced in [this paper](https://arxiv.org/pdf/2001.09977.pdf): 
- Sensibleness: Crowdsourcers were asked whether model's response makes sense given the context
- Specificity: Crowdsourcers were asked whether model's response is specific for given context, in other words we don't want our model to give general and boring responses
- SSA which is the average of two metrics above (Sensibleness Specificity Average)

|                                                     |   sensibleness |   specificity |   SSA |
|:----------------------------------------------------|---------------:|--------------:|------:|
| [tinkoff-ai/ruDialoGPT-small](https://huggingface.co/tinkoff-ai/ruDialoGPT-small)  |           0.64 |          0.5  | 0.57  |
| [tinkoff-ai/ruDialoGPT-medium](https://huggingface.co/tinkoff-ai/ruDialoGPT-medium) |           0.78 |          0.69 | 0.735 |


How to use:

```python
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
inputs = tokenizer('@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела? @@ВТОРОЙ@@', return_tensors='pt')
generated_token_ids = model.generate(
    **inputs,
    top_k=10,
    top_p=0.95,
    num_beams=3,
    num_return_sequences=3,
    do_sample=True,
    no_repeat_ngram_size=2,
    temperature=1.2,
    repetition_penalty=1.2,
    length_penalty=1.0,
    eos_token_id=50257,
    max_new_tokens=40
)
context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
context_with_response
```