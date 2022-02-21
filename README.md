# GPT-2 Fine-tuning in Vietnamese Six Eight Poems
## Model description
This is a Vietnamese GPT-2 model Six Eight Poet Model which is trained on the 10mb of Six Eight poems dataset, based on the Vietnamese Wiki GPT2 pretrained model (https://huggingface.co/danghuy1999/gpt2-viwiki)

## Purpose
This model was made only for fun and experimental study

## Dataset
The dataset is about 10k lines of Vietnamese Six Eight poems

## Result
- Train Loss: 2.7
- Val loss: 4.5



## How to use
You can use this model to generate Six Eight poems given any starting words

## Example
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("tuanle/GPT2_Poet")
model = AutoModelForCausalLM.from_pretrained("tuanle/GPT2_Poet").to(device)
text = "hỏi rằng nàng"
input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
min_length = 60
max_length = 100
sample_outputs = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id,
                                   do_sample=True,
                                   max_length=max_length,
                                   min_length=min_length,
                                #    temperature = .8,
                                #    top_k= 100,
                                   top_p = 0.8,
                                   num_beams= 10,
                                #    early_stopping=True,
                                   no_repeat_ngram_size= 2,
                                   num_return_sequences= 3)
for i, sample_output in enumerate(sample_outputs):
    print(">> Generated text {}\n\n{}".format(i+1, tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)))
    print('\n---')
```

## Demo
- Input: "hỏi rằng nàng"
- Output:\
hỏi rằng nàng đã nói ra\
cớ sao nàng lại hỏi han sự tình\
vân tiên nói lại những lời\
thưa rằng ở chốn am mây một mình\
từ đây mới biết rõ ràng\
ở đây cũng gặp một người ở đây\
hai người gặp lại gặp nhau\
thấy lời nàng mới hỏi tra việc này\
nguyệt nga hỏi việc bấy lâu\
khen rằng đạo sĩ ở đầu cửa thiền
