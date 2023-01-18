# stage 1
python3 roc.py \
	--ytrue data/ffhq_1bit_0.05/120000.ytrue.txt \
	--yscore data/ffhq_1bit_0.05/120000.yscores.txt \
	--title 'Stage 1: FFHQ' --color crimson \
	--save stage1-roc-ffhq.pdf
python3 roc.py \
	--ytrue data/imagenet_1bit_1.0/280000.ytrue.txt \
	--yscore data/imagenet_1bit_1.0/280000.yscores.txt \
	--title 'Stage 1: ImageNet' --color crimson \
	--save stage1-roc-imagnet.pdf
# stage 2
python3 roc.py \
	--ytrue data/ffhq_1bit_0.05_adm/output.ytrue.txt \
	--yscore data/ffhq_1bit_0.05_adm/output.yscores.txt \
	--title 'Stage 2: ADM' --color slateblue \
	--save stage2-roc-adm.pdf
python3 roc.py \
	--ytrue data/ffhq_1bit_0.05_ldm/img.ytrue.txt \
	--yscore data/ffhq_1bit_0.05_ldm/img.yscores.txt \
	--title 'Stage 2: LDM' --color slateblue \
	--save stage2-roc-ldm.pdf
