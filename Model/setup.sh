pip install vllm
pip install radon
pip install peft
pip install openai
pip install bitsandbytes loralib --upgrade
pip install uuid
pip install tensorboardX
pip install accelerate
pip install evaluate
pip3 install -U click
pip3 install scipy
pip3 install datasets
pip install sentence-transformers
pip install matplotlib
pip install trl
pip install einops
pip install voyageai
pip install -U git+https://github.com/huggingface/transformers.git 
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git
apt-get update
apt-get install sudo
sudo apt-get install git-lfs
sudo apt-get install unzip
sudo apt-get install htop
sudo apt-get install nano
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
export AWS_ACCESS_KEY_ID=AKIAU6GDU6QKYDI55MJV
export AWS_SECRET_ACCESS_KEY=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
export AWS_DEFAULT_REGION=us-east-2
aws s3 cp --recursive s3://codegen-project-bucket/Datasets/ /workspace/CS762_Project/Datasets
aws s3 cp --recursive s3://codegen-project-bucket/Ablations/ /workspace/CS762_Project/Ablations
# for openai
sudo apt install python3.12 
# git clone https://huggingface.co/meta-llama/Llama-2-13b-hf
# cd Llama-2-13b-hf
# rm *.safetensors
# rm model.safetensors.index.json
# git-lfs install
# git lfs pull

# cd /workspace/CS762_Project
# git clone https://huggingface.co/WhereIsAI/UAE-Large-V1
# cd UAE-Large-V1
# git-lfs install
# git lfs pull
# cd /workspace/CS762_Project

# cd /workspace/CS762_Project
# git clone https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
# cd deepseek-coder-6.7b-base
# git-lfs install
# git lfs pull
# cd /workspace/CS762_Project

# cd /workspace/CS762_Project
# git clone https://huggingface.co/microsoft/phi-2
# cd phi-2
# git-lfs install
# git lfs pull
# cd /workspace/CS762_Project

# cd /workspace/CS762_Project
# git clone https://huggingface.co/codellama/CodeLlama-7b-hf
# cd CodeLlama-7b-hf
# git-lfs install
# git lfs pull
# cd /workspace/CS762_Project

# git clone https://huggingface.co/codellama/CodeLlama-34b-hf
# cd CodeLlama-13b-hf
# git-lfs install
# git lfs pull

# git clone https://huggingface.co/codellama/CodeLlama-13b-hf
# cd CodeLlama-13b-hf
# git-lfs install
# git lfs pull

# git clone https://huggingface.co/intfloat/e5-base-v2
# cd e5-base-v2
# git-lfs install
# git lfs pull
# cd /workspace/CS762_Project

# git clone https://huggingface.co/Salesforce/codegen-350M-mono
# cd codegen-350M-mono
# git-lfs install
# git lfs pull

# git clone https://huggingface.co/intfloat/e5-large-v2
# cd e5-large-v2
# git-lfs install
# git lfs pull
# cd /workspace/CS762_Project

git config --global user.email "karthiksuresh324@gmail.com"
git config --global user.name "Karthik Suresh"