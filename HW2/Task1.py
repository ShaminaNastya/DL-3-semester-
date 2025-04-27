import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def greedy_decode(model, tokenizer, input_text, max_new_tokens=1000, eos_token_id=151645):
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
    generated = input_ids.clone()
    for i in range(max_new_tokens):
        logits = model(input_ids=generated).logits
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == eos_token_id:
            break
    return tokenizer.decode(generated[0], skip_special_tokens=False)

input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
input_text_json = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

story = greedy_decode(model, tokenizer, input_text_hedgehog)
json_result = greedy_decode(model, tokenizer, input_text_json)

print("Сказка про ёжика:")
print(story)
print("\nСгенерированный JSON:")
print(json_result)
