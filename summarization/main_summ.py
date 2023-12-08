from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from rouge import Rouge
dataset = load_dataset("cnn_dailymail", '3.0.0')
print(dataset['train'][0].keys())
stories = []
summaries = []
train_set = dataset['train']
for item in train_set:
    stories.append(item['article'])
    summary = item['highlights']
    summaries.append(summary)
print(len(stories))
print(len(summaries))

max_stor = 0
max_sum = 0
max_stor_str = ""
max_sum_str = ""

final_stories = []
final_summ = []
for i in range(len(stories)):
    stor_len = len(stories[i].split())
    sum_len = len(summaries[i].split())
    if (stor_len < sum_len):
        continue
    else:
        if stor_len < 495:
            final_stories.append(stories[i])
            final_summ.append(summaries[i])
            if len(final_stories) > 35000:
                break
print(len(final_stories), len(final_summ))

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
        max_length = 500-3
        for story in tqdm(self.stor_list, total = len(self.stor_list)):
            #input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
            encoded_story = tokenizer.encode(story,return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
            prompt = [0, 1, 2]
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
rouge_list = []
# make train and val dataloader
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=5)
val_loader = DataLoader(val_dataset, batch_size=1)
import rouge
n_epochs = 100
from model_summ import Prompt_GPT
summ_model = Prompt_GPT(config)
summ_model = summ_model.to(device)
optimizer = torch.optim.Adam(summ_model.parameters(), lr=0.01)
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
        # use output.logits and calculate the rouge score against the labels
        output = outputs.logits
        out_preds = []
        # convert summ and output to lists, as they are of batch size 1
        summ = summ.tolist()
        # argmax outpit and get the classes
        output = output.argmax(dim=-1)
        output = output.tolist()
        #print(output)
        out_preds = output[0]
        summ = summ[0]
        # find the first index i where summ[i] == 50256
        indices = [i for i, x in enumerate(summ) if x == 50256]
        index = indices[0]
        #rint(out_preds)
        summ = summ[:index]
        output = out_preds[:index]
        print(type(summ), type(output))
        summ_str = " ".join(map(str, summ))
        output_str = " ".join(map(str, output))
        rouge = Rouge()
        scores = rouge.get_scores(output_str, summ_str)
        rouge_1_recall = scores[0]['rouge-1']['r']
        loss = outputs.loss
        val_los_list.append(loss.item())
        rouge_list.append(rouge_1_recall)
        clear_cuda_memory()
        print("Epoch: ", epoch, "Batch: ", i, "Loss: ", loss.item(), "Rouge: ", rouge_1_recall)
        if i%1 ==0:
            clear_cuda_memory()
            clear_cuda_memory()
    torch.save(model, "prompt_gpt25_2.pt")
    torch.save(train_loss_list, "train_list5_2.pt")
    torch.save(val_los_list, "val_list5_2.pt")
    torch.save(rouge_list, "rouge_list5_2.pt")
    print("Model Saved")
    scheduler.step()
    clear_cuda_memory()
    clear_cuda_memory( )