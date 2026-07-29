apt install -y nvtop tree
git clone https://github.com/ivlcic/cannopy.git
cd cannopy
uv sync
source .venv/bin/activate

./data download ner
./data prepare ner

./sdjt-multi8-sweep.sh 2611

./sdjt-multi8-sweep.sh 4760

./sdjt-multi8-sweep.sh 6390


./sdjt-train.sh 2611 mm-bert
./sdjt-train.sh 2611 mdeberta3
./sdjt-train.sh 2611 xlmr


./sdjt-train.sh 4760 mm-bert
./sdjt-train.sh 4760 mdeberta3
./sdjt-train.sh 4760 xlmr

./sdjt-train.sh 6390 mm-bert
./sdjt-train.sh 6390 mdeberta3
./sdjt-train.sh 6390 xlmr



