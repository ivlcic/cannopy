apt install -y nvtop tree
git clone https://github.com/ivlcic/cannopy.git
cd cannopy
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-cu129.txt
pip install flash_attn==2.8.3 --no-build-isolation
./data download ner
./data prepare ner
./data split ner
./data analyze ner
./data resample ner-sdjt