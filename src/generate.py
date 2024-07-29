import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(model_path, tokenizer_path, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))

    # inputs = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True, truncation=True)
    # ids = inputs['input_ids'].to(device)
    # attention_mask = inputs['attention_mask'].to(device)

    ids = tokenizer.encode(f'{prompt}', return_tensors='pt')

    outputs = model.generate(
        ids,
        #attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1, 
        top_k=30,        
        top_p=0.9,      
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text