# CS762 Final Project: K-Specialist Approach to Code Generation

**Authors:** \
\
Karthik Suresh (ksuresh6@wisc.edu)\
Hafeez Ali Anees Ali (aneesali@wisc.edu)\
Calvin Kranig (ckranig@wisc.edu)

Please click [here](https://drive.google.com/file/d/1ZcJ-SixqcS4sSS8PtZCU-i7z79vDHf__/view) or open `report.pdf` in this repository to take a look at our paper!

## Abstract

We introduce a new model training and inference pipeline involving K-means clustering and a novel dataset to perform SFT (Supervised Fine Tuning) for code generation. We adopt several techniques to generate data samples synthetically, emphasizing data quality and complexity. Benchmarking our dataset against other SFT datasets for code generation, we find ours has the highest complexity scores, as evidenced by higher Cyclomatic and Halstead Complexity measures. Using a novel procedure, we train phi-2 (2.7B) and CodeLlama-Python-7B. We leveraged our collected data to train a K-means clustering model using embeddings from a Sentence Embedding model. By training the K-means model on embeddings from our collected SFT dataset, we can split the SFT data into K splits. We use these K data splits to train K LoRA adapters, acting as K experts on the given semantic cluster of questions. Using our method, our best model, phi-2 (K=10), achieves **53.54%** *pass@1* on the HumanEval benchmark, comparable to the _pass@1_ performance of larger parameter models. As we increase the K value, we also see an increase in coding performance.

<!-- 

# Synthetic Data Generation

![Screen Shot 2024-06-05 at 4 06 31 PM](https://github.com/karths8/K-Specialist-Approach-to-Code-Generation/assets/47289950/e29a16d4-952f-40b0-bd1d-bd545b2f1b08)

Three Data Generation Strategies we prototyped for our training dataset. **Strategy 1** resulted in many cases of model refusal to use the given
keyword list to produce a meaningful question-code pair. **Strategy 2** generated data points similar to the in-context examples
with slight modifications. **Strategy 3**, a combination of **Strategy 1** and **Strategy 2**, generated complex and
unique data points.

-->




## Model Training

![Screen Shot 2024-06-05 at 4 05 14 PM](https://github.com/karths8/K-Specialist-Approach-to-Code-Generation/assets/47289950/fc471e6a-d467-4980-a8cf-ab00a6b023ce)

Above is the training pipeline we followed to train our model. The Training dataset we used corresponds to the dataset we collected, as
detailed in Section 2. The PLM (Pretrained Language model), which we use as our base for training, is one of two models: The phi-2
model (2.7B) and CodeLlama-7B Python variant. The Sentence Embedding model we use is e5-base-v2
. The K different ∆W’s correspond to K different LoRA weight updates.


## Results

We achieved a **53.54%** *pass@1* rate using phi-2 2.7B when **K=10** and training on the dataset we had collected. This is comparable to many of the higher parameter code generation models in terms of coding performance as assessed by the [HumanEval benchmark]([url](https://github.com/openai/human-eval)) 

| Model                   | pass@1 | pass@10 |
|-------------------------|--------|---------|
| **Closed-Source Models** |        |         |
| GPT-3.5                 | 48.10% | -       |
| GPT-4                   | **67.00%** | -       |
| PaLM-Coder              | 36.00% | -       |
| Codex                   | 28.8%  | 46.8%   |
| **Open-Source Models**   |        |         |
| CodeLlama-Python-7B     | 38.4%  | 70.3%   |
| CodeLlama-Python-13B    | 43.3%  | 77.4%   |
| CodeLlama-Python-34B    | 53.7%  | 82.8%   |
| Llama-2-70B             | 29.9%  | -       |
| CodeT5+                 | 29.3%  | -       |
| StarCoder Prompted      | 40.8%  | -       |
| phi-2                   | 48.00%| -       |
| **Ours**                 |        |         |
| phi-2 **(K=1)**             | 49.27% | 70.97%  |
| phi-2 **(K=5)**            | 50.00% | 71.81%  |
| phi-2 **(K=10)**           | <ins>53.54%</ins> | 70.77%  |
| CodeLlama-Python-7B **(K=1)**| 41.95%| 68.33%  |
| CodeLlama-Python-7B **(K=5)**| 44.63%| 69.40%  |
| CodeLlama-Python-7B **(K=10)**|42.44%| 69.50%  |

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


The following sub-heading refers to a folder in the codebase, and following that will be a description of what we used it for and details about the files within

## Data

The dataset we collect with the collection process as detailed in the paper is stored in `collected_data.json`

<br/>

## Data Generation

These are the files that we used to prompt GPT-3.5

`generate_new_examples_gpt.py` has code that prompts GPT-3.5 for a response. 

`multiprocess_generate.py` leverages multiprocessing capabilities to spawn multiple requests to GPT-3.5 on multiple threads. This helped us in getting synthetic data at good speeds. Care must be taken to ensure that the rate limits on the requests to the OpenAI servers are not crossed.
<br/>
## Data Processing

The files here are used to process the data into types and styles we may need during data generation, training, and inferencing. 

`prepare_data.py` is used to prepare data from a `json` file for training. We use the [Huggingface Chat Templating](https://huggingface.co/docs/transformers/main/en/chat_templating) capabilities to tailor the prompt style for each of the models we train 

`prepare_data_human_eval.py` is used to prepare the [HumanEval](https://github.com/openai/human-eval) data
<br/>
## Training

Files related to training the models are kept here

`kmeans-fit.py` is used to train a K-means Clustering model given the input training data

`train_model.py` handles the training of the phi-2 and CodeLlama models by leveraging [QLoRA](https://github.com/artidoro/qlora)

`merge_lora.py` is used as and when we need to merge the LoRA adapter weights to the base PLM (Pretrained Language Model)
<br/>
## Inference

`inference.py` is used to perform inference given a model path

`vllm_inference.py` leverages the ultrafast inferencing capabilities of the [vLLM engine](https://github.com/vllm-project/vllm) to produce generations



where K is one of {1, 5, 10} and temp is one of {0.2, 0.8} and M is one of {'phi-2', 'Codellama-Python-7B'}
