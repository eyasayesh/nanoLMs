#hugging face testing grounds

import psutil
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer, GPTNeoXTokenizerFast
from tqdm import tqdm
import multiprocessing
from itertools import chain

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(device)

num_proc = multiprocessing.cpu_count()
seed, buffer_size = 42, 10_000

#tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')
tokenizer2 = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-360M')

dataset = load_dataset('HuggingFaceTB/smollm-corpus',"cosmopedia-v2", split='train', streaming=True,trust_remote_code=True)
print(dataset)
print()
iterable_dataset = iter(dataset)
print(next(iterable_dataset))
print()
dataset = dataset.shuffle(seed, buffer_size=buffer_size)
print(next(iterable_dataset))
print()
print(next(iterable_dataset))
print()

def tokenize(examples):
    examples["token_ids"] = tokenizer(
       examples["text"], truncation=False, max_length=tokenizer.model_max_length
    )
    return examples


tokenized_dataset = dataset.map(tokenize, batched=True)#, remove_columns=["text"]
print(tokenized_dataset.features)
print()
print(tokenized_dataset)
print()
tokenized_dataset = tokenized_dataset.with_format("torch")
#print(next(iter(tokenized_dataset)))
print()
print(tokenizer)
print(tokenizer2)
# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_dataset = tokenized_dataset.map(group_texts, batched=True)
# shuffle dataset
tokenized_dataset = tokenized_dataset.shuffle(seed=34)
#print(next(iter(tokenized_dataset)))
print()
"""
dataloader = DataLoader(tokenized_dataset, batch_size=32)


model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
model.train().to(device)

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
for epoch in range(3):
    dataset.set_epoch(epoch)
    for i, batch in enumerate(tqdm(dataloader, total=5)):

        print(batch)
        
        if i == 5:
            break
        #batch = {k: v.to(device) for k, v in batch.items()}
        batch_in = batch['input_ids'].to(device)
        
        outputs = model(batch_in)
        loss = outputs[0]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            print(f"loss: {loss}")
"""