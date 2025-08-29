# Financial Statement Classification API on AWS

This ML Engineering / MLOps project aims to deploy an NLP model using AWS cloud services for inference using an API. The model provides reasonable sentiment extraction from news sources such as [Reuters](https://www.reuters.com/), [Bloomberg](https://www.bloomberg.com/), [Financial Times](https://www.ft.com/), etc... 

![](https://github.com/ConsciousMachines/News-Headline-Classification/blob/main/img/diagram.svg)

# Machine Learning Model

The model used in this project is DistilBERT, fine-tuned on the [Financial Phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) dataset. The goal of the dataset is to present ~5,000 financial phrases and give them a sentiment of neutral, negative, or positive. The dataset is divided into parts depending on how many of the annotators agreed on the sentiment of the phrase, for example 50%, 75%, or 100%. This model was trained on the 75% dataset (with ~3,500 samples). Looking through the phrases, one may think some of the output labels may be subjective. Nevertheless, the model achieves 94% accuracy on the 20% validation set. Further work may involve expanding the training data and automating the labeling using some LLM.

# Deployment

Model deployment was done using Amazon SageMaker. The SageMaker endpoint was connected to a Lambda Function, which was in turn connected to an API Gateway endpoint. The API Gateway endpoint was tested using Postman. This allows any web framework to get inference from the model. 

![](https://github.com/ConsciousMachines/News-Headline-Classification/blob/main/img/post.png)

# Load Testing

To analyze the cost vs latency efficiency of model throughput, Amazon's load testing provides a fantastic way to compare different instances. Dependiing on the desired inference frequency and model latency, a cost optimized instance can be selected to host the model. As the number of requests scales, so does memory, CPU, and GPU usage, affecting the output latency.

![](https://github.com/ConsciousMachines/News-Headline-Classification/blob/main/img/load_test.png)

