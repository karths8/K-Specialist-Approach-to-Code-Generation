# CS762 Final Project: K-Specialist Approach to Code Generation

**Authors:** Karthik Suresh (ksuresh6@wisc.edu), Hafeez Ali Anees Ali (aneesali@wisc.edu), Calvin Kranig (ckranig@wisc.edu)

## Abstract

We introduce a new model training and inference pipeline involving the use of K-means clustering and a novel dataset to perform SFT (Supervised Fine Tuning) for code generation. We adopt a number of techniques to generate data samples synthetically with an emphasis on data quality and complexity. Benchmarking our dataset against other SFT datasets for code generation, we find that ours has the highest complexity scores, as evidenced by higher Cyclomatic and Halstead Complexity measures, while underperforming on the Diversity benchmark. During the training phase, we train phi-2 (2.7B) and CodeLlama-Python-7B using a novel procedure. We leveraged our collected data to train a K-means clustering model using embeddings from a Sentence Embedding model. By training the K-means model on embeddings from our collected SFT dataset, we are able to split the SFT data into K splits. We use these K data splits to train K LoRA adapters. Using our method, our best model, phi-2 (K=10), achieves **53.54%** *pass@1* on the HumanEval benchmark, which is comparable to the _pass@1_ performance of the CodeLlama-Python-34B variant. Moreover, we see an increase in performance as we increase K while keeping the number of data points the same. 

## Information on Code

The following sub-heading refer to a folder in the codebase and following that will be a description of what we used is for and details about the files within

### Data

The dataset we collect with the collection process as detailed in the paper is stored in `collected_data.json`

### Data Generation

These are the files that we used to prompt GPT-3.5

`generate_new_examples_gpt.py` has code that prompts GPT-3.5 for a response. 

`multiprocess_generate.py` leverages multiprocessing capabilities to spawn multiple requests to GPT-3.5 on multiple threads. This helped us in getting synthetic data at good speeds.

### Data Processing

The files here are used to process the data into types and styles that we may need during data generation, training and inferencing. 

`prepare_data.py` is used to prepare data from a `json` file for training. We use the [Huggingface Chat Templating](https://huggingface.co/docs/transformers/main/en/chat_templating) capabilities to tailor the prompt style for each of the models we train 

`prepare_data_human_eval.py` is used to prepare the [HumanEval](https://github.com/openai/human-eval) data

### Training

Files related to training the models are kept here

`kmeans-fit.py` is used to train a K-means Clustering model given the input training data

`train_model.py` handles the training of the phi-2 and CodeLlama models by leveraging [QLoRA](https://github.com/artidoro/qlora)

`merge_lora.py` is used as and when we need to merge the LoRA adapter weights to the base PLM (Pretrained Language Model)

### Inference

`inference.py` is used to perform inference given a model path

`vllm_inference.py` leverages the ultrafast inferencing capabilities of the [vLLM engine](https://github.com/vllm-project/vllm) to produce generations

## Reproducibility

Since running trainings or the model pipeline is not computationally feasible for reproducibility, we provide the `.jsonl` files that were generate for the HumanEval benchmark

Under `Codellama-Python-7B-humaneval-generations` and `phi-2-humaneval-generations` you can find the `.jsonl` files corresponding to the total clusters (K) and the temperature (T) in the format:

`human\_eval\_{M}\_k\_{K}_temp\_{T}` where K is one of {1, 5, 10} and temp is one of {0.2, 0.8} and M is one of {'phi-2', 'CodeLlama'}

Note that the results in our paper use a temperature of 0.2 to report the _pass@1_ accuracies and a temperature of 0.8 to report the _pass@10_ accuracies

To reproduce the scores, follow these steps:

1. Git clone and install the [human-eval](https://github.com/openai/human-eval) repository:

```
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```

You may use a conda env for this as outlined in the [human-eval](https://github.com/openai/human-eval) repo

2. If you want to test the score for a total cluster value of K and temperature of T of model M then you would run the following command:


```
$ evaluate_functional_correctness /path/to/dir/{M}-humaneval-generations/human_eval_k_{K}_temp_{T}.jsonl
```

where K is one of {1, 5, 10} and temp is one of {0.2, 0.8} and M is one of {'phi-2', 'Codellama-Python-7B'}
