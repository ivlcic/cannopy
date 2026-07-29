apt install -y nvtop tree
git clone https://github.com/ivlcic/cannopy.git
cd cannopy
uv sync
source .venv/bin/activate

./data download ner


./data prepare ner -s data.split.seed=2611 -s data.sampling.seed=2611
./sdjt-multi8-sweep.sh 2611

./data prepare ner -s data.split.seed=4760 -s data.sampling.seed=4760
./sdjt-multi8-sweep.sh 4760

./data prepare ner -s data.split.seed=6390 -s data.sampling.seed=6390
./sdjt-multi8-sweep.sh 6390


