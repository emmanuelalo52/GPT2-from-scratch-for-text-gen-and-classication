#Download Dataset
import urllib.request
import zipfile
from pathlib import Path
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from Unlabeled_data.gpt_download import download_and_load_gpt2
from Unlabeled_data.unlabeled_model import GPTModel,load_weights_into_gpt,GPTConfig
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import time

#Add LoRA
class Linear_LORA(nn.Module):
    def __init__(self,in_dim,out_dim,rank,alpha,dropout):
        super().__init__()
        self.linear = nn.Linear(in_dim,out_dim,bias=False)

        self.lora_a = nn.Linear(in_dim,rank,bias=False)
        self.lora_b = nn.Linear(in_dim,out_dim,bias=False)

        self.rank = rank
        self.alpha = alpha

        self.dropout = nn.Dropout(p=dropout)

        #freeze original weights
        self.linear.weight.requires_grad=False
        self.lora_a.weight.requires_grad=True
        self.lora_b.weight.requires_grad=True
    def forward(self,x):
        frozen_out = self.linear(x)

        lora_out = self.lora_b(self.lora_a(self.dropout(x)))

        return frozen_out + (self.alpha /self.rank) * lora_out


#dataset url
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_read_file(url,zip_path,extracted_path,data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

try:
    download_and_read_file(url, zip_path, extracted_path, data_file_path)
except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_read_file(url, zip_path, extracted_path, data_file_path)


#balance dataset
def balance_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_set = df[df['Label'] == "ham"].sample(num_spam,random_state=123)
    balanced_df = pd.concat([ham_set,df[df["Label"] == "spam"]])
    return balanced_df

#split our data
def split_train_test_val(df,train_ratio,val_ratio):
    df = df.sample(frac=1,random_state=123).reset_index(drop=True) #shuffle dataframe
    train_end = int(len(df) * train_ratio)
    validation_end = train_end + int(len(df) * val_ratio)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df,validation_df,test_df

# train_df,validation_df,test_df = split_train_test_val(balanced_df,0.7,0.1)


#Reconstruct our dataset to fit our model
class SimpleDataset(Dataset):
    def __init__(self,csv_file,tokenizer,max_length,pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        #Pretokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            #Truncate sequence if they are longer than amx_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
            #Pad sequence of longest text length
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return(torch.tensor(encoded,dtype=torch.long),
               torch.tensor(label,dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    

def cal_accuracy_loader(dataloader,model,device,num_batches=None):
    model.eval()
    correct_predictions,num_examples = 0,0
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches,len(dataloader))
    for i, (input_batch,target_batch) in enumerate(dataloader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:,-1,:]
            predicted_labels = torch.argmax(logits,dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += ((predicted_labels == target_batch).sum().item())
        else:
            break
    return correct_predictions / num_examples

def cal_loss_batch(input_batch,target_batch,model,device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,-1,:]
    loss = nn.functional.cross_entropy(logits,target_batch)
    return loss

def cal_loss_loader(dataloader,model,device,num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("Nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch,target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = cal_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss = cal_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
#Train model for classification
"""Args:Model,train dataloader, validation dataloader, number of epochs, optimizer, eval_freq,eval_iter,device"""
def train_classifier(model,train_loader,val_loader,device,optimizer,num_epochs,eval_freq,eval_iter):
    train_losses,val_losses,train_acc,val_acc = [],[],[],[] #Initalize list to track loss and accuracy
    examples_seen,global_step = 0,-1
    for epochs in range(num_epochs): #main loop
        model.train() #training mode
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad() #reset loss gradient
            loss = cal_loss_batch(input_batch,target_batch,model,device)
            loss.backward() #calculate gradient 
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] #track examples instead of tokens
            global_step += 1
            #Evaluation step
            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epochs {epochs+1} (Step {global_step:06d}): "
                      f"Train loss: {train_loss:.3f},"
                      f"Validation loss: {val_loss:.3f}")
                
        #Calculate Accuracy after each epochs
        train_accuracy = cal_accuracy_loader(train_loader,model,device,num_batches=eval_iter)
        val_accuracy = cal_accuracy_loader(val_loader,model,device,num_batches=eval_iter)

        print(f"Training Accuracy: {train_accuracy*100:.2f}% |", end="")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
    return train_losses,val_losses,train_acc,val_acc,examples_seen

def replace_linear_with_lora(model,rank,alpha,dropout,alternative=False):
    for name,module in model.named_children():
        if isinstance(module,nn.Linear):
            if alternative:
                setattr(model,name,Linear_LORA(module.in_features,module.out_features,rank,alpha,dropout))
        else:
            replace_linear_with_lora(module,rank,alpha,dropout,alternative)
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )    
    # Alternatively, a more robust version is the following one, which handles the max_length=None case better
    # max_len = min(max_length,supported_context_length) if max_length else supported_context_length
    # input_ids = input_ids[:max_len]
    
    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"

def plot_vlaue(epoch_seen,examples_seen,train_values,val_values,label="loss"):
    fig,ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epoch_seen,train_values,label=f"Training {label}")
    ax1.plot(epoch_seen,val_values,linestyle="-.",label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen,train_values,alpha=0)
    ax2.set_xlabel("Example seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
# -------------------------------------------------------------------------- Fine-tuning ------------------------------------------------------------------------------------------
#Fine tune model
if __name__ == "__main__":
    # File paths
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    # Download dataset
    try:
        download_and_read_file(url, zip_path, extracted_path, data_file_path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_read_file(backup_url, zip_path, extracted_path, data_file_path)

    # Read and process
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = balance_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, val_df, test_df = split_train_test_val(balanced_df, 0.7, 0.1)

    train_df.to_csv("train.csv", index=None)
    val_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    # Datasets
    train_dataset = SimpleDataset("train.csv", tokenizer, max_length=None)
    val_dataset = SimpleDataset("validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset = SimpleDataset("test.csv", tokenizer, max_length=train_dataset.max_length)

    # DataLoaders
    batch_size = 8
    num_workers = 0

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    for input_batch, target_batch in train_dataloader:
        break  # Just take the first batch

    print("input dimension: ", input_batch.shape)
    print("target dimension: ", target_batch.shape)
    print(f"{len(train_dataloader)} training batches")
    print(f"{len(val_dataloader)} validation batches")
    print(f"{len(test_dataloader)} test batches")

    # Fine-tuning config
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


    model = GPTModel(GPTConfig(**BASE_CONFIG))
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size,models_dir="gpt2")
    load_weights_into_gpt(model,params)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    lora_rank = 8
    lora_alpha = 8
    lora_dropout = BASE_CONFIG["drop_rate"]
    replace_linear_with_lora(model,rank=lora_alpha,alpha=lora_alpha,dropout=lora_dropout,alternative=False)
    #freeze the model
    for params in model.parameters():
        params.requires_grad=False
    
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"],out_features=num_classes)
    model.to(device)

    #unfreeze lora layer
    for name,params in model.named_parameters():
        if "lora" in name or "out_head" in name:
            params.requires_grad=True
    for params in model.trf_blocks[-1].parameters():
        params.requires_grad=True
    for params in model.final_norm.parameters():
        params.requires_grad=True

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.1)
    num_epochs = 5
    train_losses,val_losses,train_acc,val_acc,examples_seen = train_classifier(model,train_dataloader,val_dataloader,device,optimizer,num_epochs=num_epochs,eval_freq=50,eval_iter=5)
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Training completed in {execution_time:.2f} minutes")

    #loss
    epochs_seen_loss = torch.linspace(0,num_epochs,len(train_losses))
    example_seen_tensor = torch.linspace(0,examples_seen,len(train_losses))

    plot_vlaue(epochs_seen_loss,example_seen_tensor,train_losses,val_losses)

    #Classification accuracy
    epochs_seen = torch.linspace(0,num_epochs,len(train_acc))
    example_seen_tensor_accs = torch.linspace(0,examples_seen,len(train_acc))

    plot_vlaue(epochs_seen,example_seen_tensor_accs,train_acc,val_acc,label="Accuracy")
    # text_2 = (
    # "You are a winner you have been specially"
    # " selected to receive $1000 cash or a $2000 award." 
    # )
    # print(classify_review(text_2,model,tokenizer,device,max_length=train_dataset.max_length))


    # with torch.no_grad():
    #     train_loss = cal_loss_loader(train_dataloader,model,device,num_batches=5)
    #     val_loss = cal_loss_loader(val_dataloader,model,device,num_batches=5)
    #     test_loss = cal_loss_loader(test_dataloader,model,device,num_batches=5)
    # print(f"Training loss: {train_loss:.3f}")
    # print(f"Validation loss: {val_loss:.3f}")
    # print(f"Test loss: {test_loss:.3f}")
    # train_accuracy = cal_accuracy_loader(
    #     train_dataloader, model, device, num_batches=10
    # )
    # val_accuracy = cal_accuracy_loader(
    #     val_dataloader, model, device, num_batches=10
    # )
    # test_accuracy = cal_accuracy_loader(
    #     test_dataloader, model, device, num_batches=10
    # )
    # print(f"Training accuracy: {train_accuracy*100:.2f}%")
    # print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    # print(f"Test accuracy: {test_accuracy*100:.2f}%")
    #to test
    # inputs = tokenizer.encode("Do you have time")
    # inputs = torch.tensor(inputs).unsqueeze(0)
    # print("Inputs: ", inputs)
    # print("Inputs shape: ", inputs.shape)
    # #to use our last layer (classes)
    # with torch.no_grad():
    #     outputs = model(inputs)
    # print("Outputs:\n", outputs)
    # print("Outputs dimension: ", outputs.shape)
    # print(model)

    # text1 = ("Is the following text 'spam'? Answer with 'yes' or 'no':"
    # " 'You are a winner you have been specially"
    # " selected to receive $1000 cash or a $2000 award.'")
    # token_ids = generate_text(model=model,idx=text_to_tokens(text1,tokenizer=tokenizer),max_new_tokens=23,context_size=BASE_CONFIG["context_length"])
    # print(token_to_text(token_ids=token_ids,tokenizer=tokenizer))