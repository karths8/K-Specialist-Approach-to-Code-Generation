pip install vllm
pip install lexicalrichness
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
pip install -U git+https://github.com/huggingface/transformers.git 
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git
apt-get update
apt-get install sudo
sudo apt-get install git-lfs
sudo apt-get install unzip
sudo apt-get install htop
sudo apt-get install nano
# for openai
sudo apt install python3.12 

git clone https://huggingface.co/microsoft/phi-2
cd phi-2
git-lfs install
git lfs pull

cd /workspace/CS762_Project
git clone https://huggingface.co/codellama/CodeLlama-7b-Python-hf
cd CodeLlama-7b-Python-hf
git-lfs install
git lfs pull


git clone https://huggingface.co/intfloat/e5-base-v2
cd e5-base-v2
git-lfs install
git lfs pull
