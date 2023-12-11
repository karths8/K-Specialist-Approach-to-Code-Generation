apt-get update
apt-get install sudo
sudo apt-get install git-lfs
sudo apt-get install unzip
pip install peft
pip install bitsandbytes loralib --upgrade
pip install uuid
pip install accelerate
pip install evaluate
pip3 install -U click
pip3 install scipy
pip3 install datasets
pip install sentence-transformers
pip install trl
pip install -U git+https://github.com/huggingface/transformers.git 
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git

# git clone https://huggingface.co/meta-llama/Llama-2-13b-hf
# cd Llama-2-13b-hf
# rm *.safetensors
# rm model.safetensors.index.json
# git-lfs install
# git lfs pull

# git clone https://huggingface.co/meta-llama/Llama-2-13b-hf
# cd Llama-2-13b-hf
# git-lfs install
# git lfs pull

# git clone https://huggingface.co/codellama/CodeLlama-7b-Python-hf
# cd CodeLlama-7b-Python-hf
# git-lfs install
# git lfs pull

# git clone https://huggingface.co/codellama/CodeLlama-34b-Python-hf
# cd CodeLlama-34b-Python-hf
# git-lfs install
# git lfs pull

git clone https://huggingface.co/intfloat/e5-base-v2
cd e5-base-v2
git-lfs install
git lfs pull

git clone https://huggingface.co/intfloat/e5-large-v2
cd e5-large-v2
git-lfs install
git lfs pull


# git config --global user.email "karthiksuresh324@gmail.com"
# git config --global user.name "Karthik Suresh"