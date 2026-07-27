apt install -y nvtop tree
git clone https://github.com/ivlcic/cannopy.git
cd cannopy
uv sync
./data download ner
./data prepare ner
