import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

def beam_search_generate(model, tokenizer, input_text, num_beams=4, length_penalty=1.0, max_new_tokens=1000, eos_token_id=151645, device='cpu'):
    model.eval()
    encoding = tokenizer(input_text, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    Candidate = lambda tokens, score, finished: {'tokens': tokens, 'score': score, 'finished': finished}

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        logprobs = F.log_softmax(logits, dim=-1)
        topk_logprobs, topk_indices = torch.topk(logprobs, num_beams, dim=-1)

    candidates = []
    finished = []

    for i in range(num_beams):
        next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
        tokens = torch.cat([input_ids, next_token], dim=-1)
        score = topk_logprobs[0, i].item()
        is_finished = (next_token.item() == eos_token_id)
        candidate = Candidate(tokens, score, is_finished)
        if is_finished:
            finished.append(candidate)
        else:
            candidates.append(candidate)

    while candidates and len(finished) < num_beams:
        all_candidates = []
        for cand in candidates:
            with torch.no_grad():
                outputs = model(input_ids=cand['tokens'], attention_mask=torch.ones_like(cand['tokens']))
                logits = outputs.logits[:, -1, :]
                logprobs = F.log_softmax(logits, dim=-1)
                topk_logprobs, topk_indices = torch.topk(logprobs, num_beams, dim=-1)
            for i in range(num_beams):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                tokens = torch.cat([cand['tokens'], next_token], dim=-1)
                score = cand['score'] + topk_logprobs[0, i].item()
                is_finished = (next_token.item() == eos_token_id)
                candidate = Candidate(tokens, score, is_finished)
                if is_finished:
                    finished.append(candidate)
                else:
                    all_candidates.append(candidate)

        all_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)
        candidates = all_candidates[:num_beams]

        if len(finished) >= num_beams:
            break

        if candidates and candidates[0]['tokens'].shape[1] - input_ids.shape[1] >= max_new_tokens:
            finished.extend(candidates)
            break

    def rank_score(cand):
        length = cand['tokens'].shape[1] - input_ids.shape[1]
        return cand['score'] / (length ** length_penalty if length_penalty != 0 else 1.0)

    if finished:
        best = max(finished, key=rank_score)
    else:
        best = max(candidates, key=rank_score)

    return best['tokens'][0]

input_text_hedgehog = (
    '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n'
    '<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n'
    '<|im_start|>assistant\n')

input_text_json = (
    '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n'
    '<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n'
    '<|im_start|>assistant\n')

params = [
    {"num_beams": 1, "length_penalty": 1.0},
    {"num_beams": 4, "length_penalty": 1.0},
    {"num_beams": 4, "length_penalty": 0.5},
    {"num_beams": 4, "length_penalty": 2.0},
    {"num_beams": 8, "length_penalty": 1.0}]

for i, p in enumerate(params, 1):
    print(f"Параметры: num_beams={p['num_beams']}, length_penalty={p['length_penalty']} ")

    output_ids = beam_search_generate(
        model, tokenizer, input_text_hedgehog,
        num_beams=p['num_beams'], length_penalty=p['length_penalty'], device=device)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    print(f"Сказка:\n{output_text}\n")

    output_ids = beam_search_generate(
        model, tokenizer, input_text_json,
        num_beams=p['num_beams'], length_penalty=p['length_penalty'], device=device)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
    print(f"JSON:\n{output_text}\n")
