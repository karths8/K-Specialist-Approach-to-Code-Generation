import os
from peft import PeftModel
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--model_dir', default='/workspace/CS762_Project/phi-2', type=str, help="tokenizer directory")
parser.add_argument('--num_clusters', default=1, type=int, help="Number of clusters")
parser.add_argument('--lora_0', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_1', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_2', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_3', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_4', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_5', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_6', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_7', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_8', default='0', type=str, help="Lora checkpoint chosen")
parser.add_argument('--lora_9', default='0', type=str, help="Lora checkpoint chosen")
args = parser.parse_args()

def merge_and_save_model(model_dir, lora_path, output_dir):
    device = "cuda"
    print('Loading model for Merging')
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', trust_remote_code=True)
    print('Loading Lora Weights')
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True)

def lora_list(loras, num_clusters):
    return {idx:'checkpoint-'+l for idx,l in enumerate(loras) if idx<num_clusters}


def main():
    lora_checkpoints = lora_list([args.lora_0,args.lora_1,args.lora_2,args.lora_3,args.lora_4,args.lora_5,args.lora_6,args.lora_7,args.lora_8,args.lora_9], args.num_clusters)
    merged_models_path = '/workspace/CS762_Project/Merged_models'
    model_name = args.model_dir.split('/')[-1]
    for k in range(args.num_clusters):
        lora_path = f'/workspace/CS762_Project/Results/{model_name}/total_clusters_{args.num_clusters}/k_{k}/{lora_checkpoints[k]}'
        merged_save_path = f'/workspace/CS762_Project/Results/{model_name}/total_clusters_{args.num_clusters}/k_{k}/merged_model'
        os.makedirs(merged_save_path, exist_ok=True)
        merge_and_save_model(args.model_dir, lora_path, merged_save_path)

if __name__=='__main__':
    main()
        
    