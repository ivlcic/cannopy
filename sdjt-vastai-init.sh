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


rsync -uva -e "ssh -p 40294" root@31.36.200.4:/workspace/cannopy/result/data/split/ner-sdjt.s2611/ ~/projects/studies/cannopy/result/data/split/ner-sdjt.s2611
rsync -uva -e "ssh -p 30842" root@192.165.134.28:/workspace/cannopy/result/data/split/ner-sdjt.s4760/ ~/projects/studies/cannopy/result/data/split/ner-sdjt.s4760
rsync -uva -e "ssh -p 45387" root@90.185.78.102:/workspace/cannopy/result/data/split/ner-sdjt.s6390/ ~/projects/studies/cannopy/result/data/split/ner-sdjt.s6390

rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.mm-bert.*.pretrain.s*" -e "ssh -p 42206" root@83.10.150.87:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.mdeberta3.*.pretrain.s*" -e "ssh -p 20123" root@87.197.119.154:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.mm-bert.*.pretrain.s*" -e "ssh -p 20055" root@87.197.119.154:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.mdeberta3.*.pretrain.s*" -e "ssh -p 30722" root@192.165.134.28:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.mm-bert.*.pretrain.s*" -e "ssh -p 7114" root@92.49.17.100:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.mdeberta3.*.pretrain.s*" -e "ssh -p 31372" root@192.165.134.28:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.xlmr.*.pretrain.s*" -e "ssh -p 40294" root@31.36.200.4:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.xlmr.*.pretrain.s*" -e "ssh -p 30842" root@192.165.134.28:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt
rsync -uva --exclude="checkpoint-*" --exclude="ner-sdjt.pretrain-multi7-full-*.xlmr.*.pretrain.s*" -e "ssh -p 45387" root@90.185.78.102:/workspace/cannopy/result/train/token/ner-sdjt/ ~/projects/studies/cannopy/result/train/token/ner-sdjt

./eval token ner-sdjt -c mm-bert
./eval token ner-sdjt -c mdeberta3
./eval token ner-sdjt -c xlmr
