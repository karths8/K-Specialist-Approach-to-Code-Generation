# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import DatasetDict, Dataset
from trl import SFTTrainer
import json

# This example fine-tunes Llama v2 model on Guanace dataset
# using QLoRA. At the end of the script we perform merging the weights
# Use it by correctly passing --model_name argument when running the
# script. 
#
# Versions used:
# accelerate == 0.21.0
# peft == 0.4.0
# bitsandbytes == 0.40.2
# transformers == 4.31.0
# trl == 0.4.7

# For models that have `config.pretraining_tp > 1` install:
# pip install git+https://github.com/huggingface/transformers.git

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    save_total_limit: Optional[int] = field(default=3)
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=512)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=256)
    max_seq_length: Optional[int] = field(default=1024)
    # num_train_epochs: Optional[int] = field(default=1)
    model_name: Optional[str] = field(
        default="/workspace/CS762_Project/CodeLlama-7b-Python-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    
    dataset_name: Optional[str] = field(
        default="/workspace/CS762_Project/generated_data",
        metadata={"help": "The preference dataset to use."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.05, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=200, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    num_clusters: int = field(default=1)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print(script_args)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return {'trainable_params':trainable_params, 'all_param':all_param, 'trainable_percent': 100 * trainable_params / all_param}


def create_and_prepare_model(script_args):
    model_name = script_args.model_name
    compute_dtype = getattr(torch, script_args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=script_args.use_4bit,
        bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=script_args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and script_args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        use_auth_token=True,
        trust_remote_code=True
    )
    
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1 
    model_d = model_name.split('/')[-1]
    if model_d=='phi-2':
        target_modules = ['Wqkv','out_proj','fc1','fc2']
    else:
        target_modules = ['gate_proj', 'down_proj', 'up_proj', 'q_proj', 'v_proj', 'k_proj', 'o_proj']
    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules = target_modules
        # target_modules = ['Wqkv']
        # target_modules = ['gate_proj', 'down_proj', 'up_proj', 'q_proj', 'v_proj', 'k_proj', 'o_proj']
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    params = print_trainable_parameters(model)
    print(f"trainable params: {params['trainable_params']} || all params: {params['all_param']} || trainable%: {params['trainable_percent']}")
    return model, peft_config, tokenizer

def train_model(script_args):
    total = script_args.num_clusters
    model_name = script_args.model_name
    model_dir = model_name.split('/')[-1]
    args = asdict(script_args)
    base_dir = f'/workspace/CS762_Project/Results/{model_dir}/total_clusters_{total}/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(base_dir+'training_metadata.json', 'w') as json_file:
        json.dump(args, json_file)
    for k in range(total):
        output_dir = base_dir+f'k_{k}'
        dataset_name = f'/workspace/CS762_Project/Prepared_data/{model_dir}/k_{total}/generated_data_k_{total}_cluster_{k}'
        # dataset_name = f'/workspace/CS762_Project/Model/k_{total}/generated_data_k_{total}_cluster_{k}'
        
        training_arguments = TrainingArguments(
            report_to='tensorboard',
            output_dir=output_dir,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            per_device_eval_batch_size=script_args.per_device_eval_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            evaluation_strategy='steps',
            eval_steps=script_args.save_steps,
            optim=script_args.optim,
            save_strategy='steps',
            save_steps=script_args.save_steps,
            logging_steps=script_args.logging_steps,
            learning_rate=script_args.learning_rate,
            fp16=script_args.fp16,
            bf16=script_args.bf16,
            max_grad_norm=script_args.max_grad_norm,
            # max_steps=script_args.max_steps,
            warmup_ratio=script_args.warmup_ratio,
            group_by_length=script_args.group_by_length,
            lr_scheduler_type=script_args.lr_scheduler_type,
            num_train_epochs = script_args.num_train_epochs,
            save_total_limit=script_args.save_total_limit,
            metric_for_best_model='eval_loss'
            
        )
        
        model, peft_config, tokenizer = create_and_prepare_model(script_args)
        model.config.use_cache = False
        # dataset = load_dataset(script_args.dataset_name, split="train")
        full_dataset = DatasetDict.load_from_disk(dataset_name)
        # Fix weird overflow issue with fp16 training
        tokenizer.padding_side = "right"
        
        trainer = SFTTrainer(
            model=model,
            train_dataset= full_dataset['train'],
            eval_dataset = full_dataset['val'],
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=script_args.max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=script_args.packing
        )
        
        trainer.train()


def main():
    # model_list = ['/workspace/CS762_Project/phi-2']
    # k_list = [1, 5, 10]
    # for model_name in model_list:
    #     for total in k_list:
    train_model(script_args)

if __name__=='__main__':
    main()