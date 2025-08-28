

import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import tarfile
import os



# prepare data
if True:
    encode_dict = {}
    def encode_cat(x):
        if x not in encode_dict.keys():
            encode_dict[x]=len(encode_dict)
        return encode_dict[x]

    # file_dir = r'C:\Users\cudac\Desktop\Sentences_AllAgree.txt'
    file_dir = r'C:\Users\cudac\Desktop\Sentences_75Agree.txt'
    with open(file_dir, 'r', encoding="latin-1") as file:
        lines = file.readlines()
    sentences = []
    sentiments = []
    for line in lines:
        # print(line)
        parts = line.split('@')
        if len(parts) != 2:
            raise Exception('more than 2 parts')
        assert len(parts) == 2
        sentence, sentiment = parts
        sentiment = sentiment.replace('\n','') # remove newline from sentiment
        sentences.append(sentence)
        sentiments.append(sentiment)

    df = pd.DataFrame({
        "Sentiment": sentiments,
        "Headline": sentences,
    })
    print(df)

    # s3_path = r'C:\Users\cudac\Desktop\all-data.csv'
    # df = pd.read_csv(s3_path, encoding='latin1', header=None, names=['Sentiment','Headline'])
    # df

    df['ENCODE_CAT']= df['Sentiment'].apply(lambda x:encode_cat(x))
    df = df.reset_index(drop=True)
    df = df.drop('Sentiment', axis=1)
    print(df)




class DBERT_finetuned(torch.nn.Module):

    def __init__(self, token_file=None):

        super().__init__()
    
        self.tokenizer  = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.l1             = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768,768)
        self.relu           = torch.nn.ReLU()
        self.dropout        = torch.nn.Dropout(0.3)
        self.classifier     = torch.nn.Linear(768,3)

    def forward(self,input_ids, attention_mask):

        output_1     = self.l1(input_ids=input_ids,attention_mask=attention_mask)
        hidden_state = output_1[0]
        x            = hidden_state[:,0]
        x            = self.pre_classifier(x)
        x            = self.relu(x)
        x            = self.dropout(x)
        x            = self.classifier(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DBERT_finetuned()
model.to(device)

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 1], dtype=torch.long)
        }

    def __len__(self):
        return self.len



# remove duplicate rows in the original data
df.size
df = df.drop_duplicates().reset_index(drop=True)

train_dataset, test_dataset = train_test_split(
    df,
    test_size=0.2,       
    random_state=200,    
)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = test_dataset.reset_index(drop=True)
print("Full dataset: {}".format(df.shape))
print("Train dataset: {}".format(train_dataset.shape))
print("Test dataset: {}".format(test_dataset.shape))
train_dataset
test_dataset


# check overlap
set(train_dataset.Headline).intersection(set(test_dataset.Headline))


MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

training_set = NewsDataset(train_dataset, model.tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, model.tokenizer, MAX_LEN)

train_parameters = {
    'batch_size':TRAIN_BATCH_SIZE,
    'shuffle':True,
    'num_workers':0
}
test_parameters = {
    'batch_size':VALID_BATCH_SIZE,
    'shuffle':True,
    'num_workers':0
}

training_loader = DataLoader(training_set, **train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)


def calculate_accuracy(big_idx,targets):
    return (big_idx==targets).sum().item()


def train(epoch, model, device, training_loader, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    # _steps = 50
    for _,data in enumerate(training_loader,0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)


        loss = loss_function(outputs,targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim = 1)
        n_correct += calculate_accuracy(big_idx, targets)

        nb_tr_steps +=1
        nb_tr_examples +=targets.size(0)

        # if _ % _steps == 0:
        #     loss_step = tr_loss/nb_tr_steps
        #     accu_step = (n_correct*100)/nb_tr_examples
        #     # print(f"Training loss per {_steps} steps: {loss_step}")
        #     print(f"Training Accuracy per {_steps} steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #print(f"The total accuracy for epoch {epoch}: {(n_correrct*100)/nb_tr_examples}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    # print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training accuracy Epoch: {epoch_accu}")

    return


def valid(epoch, model, testing_loader, device, loss_function):

    print('starting validation...')
    model.eval()

    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    with torch.no_grad():

        for _, data in enumerate(testing_loader,0):

            try:
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.long)

                outputs = model(ids, mask).squeeze()

                loss = loss_function(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim = 1)
                n_correct += calculate_accuracy(big_idx,targets)

                nb_tr_steps +=1
                nb_tr_examples += targets.size(0)

                # if _ % 1000 == 0:
                #     # loss_step = tr_loss/nb_tr_steps
                #     accu_step = (n_correct*100)/nb_tr_examples
                #     # print(f"Validation loss per 1000 steps: {loss_step}")
                #     print(f"Validation accuracy per 1000 steps: {accu_step}")

            except:
                pass

    # epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    # print(f"Validation loss per Epoch: {epoch_loss} at epoch {epoch}")
    print(f"Validation accuracy epoch: {epoch_accu} at epoch {epoch}")
    print('ending validation...')
    return


LEARNING_RATE = 1e-05
optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss()


# save directories
output_dir = r'C:\Users\cudac\Desktop'
model_base = 'model'
model_gz   = os.path.join(output_dir, f"{model_base}.tar.gz")
model_dir  = os.path.join(output_dir, model_base)


# training loop
EPOCHS = 4
for epoch in range(EPOCHS):
    print(f"starting epoch: {epoch}")
    train(epoch, model, device, training_loader, optimizer, loss_function)
    valid(epoch, model, testing_loader, device, loss_function)

    # save params
    _ = torch.save(model.state_dict(), f'{model_dir}_epoch_{epoch}') 


# save model params
with tarfile.open(model_gz, "w:gz") as tar:
    tar.add(model_dir, arcname=model_base)


# load model params
# model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
model.load_state_dict(torch.load(model_dir))





# manual testing
test_dataset.to_csv("test_dataset.csv", index=False)  
test_dataset = pd.read_csv("test_dataset.csv")

encode_dict = {0:'neutral', 1:'positive', 2:'negative'}
decode_dict = {j:i for i,j in encode_dict.items()}

x = test_dataset.Headline
y = test_dataset.ENCODE_CAT
model.eval()

num_correct = 0
num_wrong = 0

with torch.no_grad():

    for i in range(x.size):
        # print(i)

        title = str(x[i])
        title = " ".join(title.split())
        category = int(y[i])

        inputs = model.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        data = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'targets': torch.tensor(y[i], dtype=torch.long)
        }

        ids = data['ids'].unsqueeze(0).to(device, dtype = torch.long)
        mask = data['mask'].unsqueeze(0).to(device, dtype = torch.long)
        # targets = data['targets'].unsqueeze(0).to(device, dtype = torch.long)

        outputs = model(ids, mask).squeeze()

        _val, _idx = torch.max(outputs, dim=0)
        
        # check if model sentiment equals real sentiment
        _model_guess = int(_idx)
        if _model_guess == category:
            num_correct += 1
        else:
            print()
            print(f'model guess: {decode_dict[_model_guess]}')
            print(f'correct: {decode_dict[category]}')
            print(title)
            num_wrong += 1

print(num_correct / (num_correct + num_wrong))