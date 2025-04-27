import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

def nucleus_sample_generate(model, tokenizer, input_text, max_new_tokens=1000, eos_token_id=151645, temperature=1.0, top_p=0.9, device='cpu'):
    model.eval()
    encoding = tokenizer(input_text, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    generated = input_ids.clone()

    for i in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs <= top_p
            mask[..., 0] = True

            filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            next_token_in_sorted = torch.multinomial(filtered_probs, num_samples=1)
            next_token_id = sorted_indices.gather(-1, next_token_in_sorted)

        generated = torch.cat([generated, next_token_id], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id, device=device)], dim=-1)
        if (next_token_id == eos_token_id).all():
            break
    return generated[0]

input_text_hedgehog = (
    '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n'
    '<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n'
    '<|im_start|>assistant\n')

input_text_json = (
    '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n'
    '<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n'
    '<|im_start|>assistant\n')

params = [
    {"temperature": 1.0, "top_p": 0.9},
    {"temperature": 1.0, "top_p": 0.15},
    {"temperature": 0.5, "top_p": 0.9},
    {"temperature": 0.5, "top_p": 0.15}]

for i, p in enumerate(params, 1):
    print(f" Параметры: temperature={p['temperature']}, top_p={p['top_p']} ")
    output_ids = nucleus_sample_generate(
        model, tokenizer, input_text_hedgehog,
        temperature=p['temperature'], top_p=p['top_p'], device=device)
    
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    print(f"Сказка:\n{output_text}\n")
    output_ids = nucleus_sample_generate(
        model, tokenizer, input_text_json,
        temperature=p['temperature'], top_p=p['top_p'], device=device)
    
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    print(f"JSON:\n{output_text}\n")
