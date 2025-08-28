

import torch
import json
import os
from transformers import DistilBertTokenizer, DistilBertModel


class DBERT_finetuned(torch.nn.Module):

    def __init__(self):

        super().__init__()
    
        # tools
        self.device         = None
        self.tokenizer      = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.encode_dict    = {0:'neutral', 1:'positive', 2:'negative'}

        # network
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


def model_fn(model_dir):
    print("Loading model from :", model_dir)

    model = DBERT_finetuned()

    model_path = os.path.join(model_dir, "model")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    model.eval()
    model.device = torch.device("cpu")#("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model.device)
    return model


def input_fn(request_body,request_content_type):

    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        sentence = input_data['inputs']
        return sentence
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def output_fn(prediction, accept):

    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
    

def predict_fn(input_data, model):

    inputs = model.tokenizer(input_data, return_tensors="pt")
    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(ids,mask)

    probabilities = torch.softmax(outputs, dim = 1).numpy()
    predicted_class = probabilities.argmax(axis=1)[0]
    predicted_label = model.encode_dict[predicted_class]

    return {'predicted_label': predicted_label, 'probabilities':probabilities.tolist()}
