from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
# modeled on squad dataset, question answering model
dataset = load_dataset("squad_v2")
train_set = dataset['train']
final_stories = []
final_summ = []
from tqdm import tqdm
for item in tqdm(train_set, total = len(train_set), desc="Processing..."):
    question = item['question']
    context = item['context']
    if len(item['answers']['text']) < 1:
        continue
    answer = item['answers']['text'][0]
    ans_len = len(answer.split())
    ques_len = len(question.split())
    context_len = len(context.split())
    if context_len > 0 and ques_len > 0 and ans_len > 0:
        in_len = context_len + ques_len+ 1
        if in_len < 493:
            input_str = context
            input_str = input_str + " <sep> " + question
            output_str = answer
            final_stories.append(input_str)
            final_summ.append(output_str)
            if len(final_stories) > 35000:
                break
print(len(final_stories))
print(len(final_summ))
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# Tokenize the input and target text
# input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
# target_ids = tokenizer.encode(target_text, return_tensors="pt", max_length=1024, truncation=True)
# while tokenizing inputs and targets, pad to length 100
max_length = 500
import torch.nn as nn
# Encode input text
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usinf Device: ", device)
from tqdm import tqdm
class sum_dataset(torch.utils.data.Dataset):
    def __init__(self, stories_list, summaries_list):
        self.stor_list = stories_list
        self.sum_list = summaries_list
        self.stor_indexes = self.encode_stories()
        self.sum_indexes = self.encode_summaries()
    def encode_stories(self):
        stor_indexes = []
        max_length = 500-5
        for story in tqdm(self.stor_list, total = len(self.stor_list)):
            #input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
            encoded_story = tokenizer.encode(story,return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
            prompt = [0, 1, 2, 3, 4]
            # convert to tensor
            prompt = torch.tensor(prompt)
            # add batch dimension 1
            prompt = prompt.unsqueeze(0)
            encoded_story = torch.cat((prompt, encoded_story), 1)
            stor_indexes.append(encoded_story)
        return stor_indexes
    def encode_summaries(self):
        sum_indexes = []
        max_length = 500
        for summary in tqdm(self.sum_list, total = len(self.sum_list)):
            encoded_summary = tokenizer.encode(summary, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
            sum_indexes.append(encoded_summary)
        return sum_indexes
    def __len__(self):
        return len(self.stor_list)
    def __getitem__(self, index):
        return self.stor_indexes[index].to(device), self.sum_indexes[index].to(device)
dataset = sum_dataset(final_stories, final_summ)
for item in dataset:
    print(item[0].shape, item[1].shape)

# make dataloader
from torch.utils.data import DataLoader
# make train and val dataloader
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=5)
val_loader = DataLoader(val_dataset, batch_size=5)

n_epochs = 100
from model_qa import Prompt_GPT
summ_model = Prompt_GPT(config)
summ_model = summ_model.to(device)
optimizer = torch.optim.Adam(summ_model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
train_loss_list =  []
val_los_list = []
def clear_cuda_memory():
    torch.cuda.empty_cache()
    for i in range(32):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
for epoch in range(n_epochs):
    print("Epoch: ", epoch)
    summ_model.train()
    for i, (stor, summ) in enumerate(train_loader):
        stor = stor.to(device)
        summ = summ.to(device)
        # change from 4, 1, 500 to 4, 500
        stor = stor.squeeze(1)
        summ = summ.squeeze(1)
        #print(stor.shape, summ.shape)
        outputs = summ_model(input_ids = stor, labels=summ)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_list.append(loss.item())
        print("Epoch: ", epoch, "Batch: ", i, "Loss: ", loss.item())
        if i%100 ==0:
            clear_cuda_memory()
            clear_cuda_memory()
        if i%1000 ==0:
            scheduler.step()
    #summ_model.eval()
    clear_cuda_memory()
    clear_cuda_memory()
    for i, (stor, summ) in enumerate(val_loader):
        stor = stor.to(device)
        summ = summ.to(device)
        stor = stor.squeeze(1)
        summ = summ.squeeze(1)
        with torch.no_grad():
            outputs = summ_model(input_ids = stor, labels=summ)
        clear_cuda_memory()
        loss = outputs.loss
        val_los_list.append(loss.item())
        clear_cuda_memory()
        print("Epoch: ", epoch, "Batch: ", i, "Loss: ", 0)
        if i%1 ==0:
            clear_cuda_memory()
            clear_cuda_memory()
    torch.save(model, "prompt_gpt_qa_2.pt")
    torch.save(train_loss_list, "train_list_qa_2.pt")
    torch.save(val_los_list, "val_list_qa_2.pt")
    print("Model Saved")
    scheduler.step()
    clear_cuda_memory()
    clear_cuda_memory()

    