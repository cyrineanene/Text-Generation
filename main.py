import re
from src.generate import generate_text
from src.train import train
from src.load_data import load_dataset, load_data_collator, read_documents_from_directory

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

#Load your data 
directory = 'input/'
text_data = read_documents_from_directory(directory)
text_data = re.sub(r'\n+', '\n', text_data).strip()

#train the model
train_file_path = 'input/snow_white.txt'
model_name = 'gpt2'
output_dir = "./output/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
train_dataset = load_dataset(train_file_path, tokenizer)
data_collator = load_data_collator(tokenizer)
train(
    output_dir=output_dir,
    train_dataset = train_dataset,
    data_collator = data_collator,
    model = model,
    tokenizer = tokenizer
)

#Generate your text
model_path = './pretrained/model'
tokenizer_path = './pretrained/tokenizer'
prompt = input('enter your prompt: ')

generated_text = generate_text(model_path, tokenizer_path, prompt)

with open("generated_text.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)
