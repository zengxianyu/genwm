python3 roc.py \
	--ytrue data/ffhq_1bit_0.05/120000.ytrue.txt \
	--yscore data/ffhq_1bit_0.05/120000.yscores.txt \
	--title 'Stage 1: ROC for $W$' --color crimson \
	--save stage1-roc.pdf
python3 roc.py \
	--ytrue data/ffhq_1bit_0.05_adm/output.ytrue.txt \
	--yscore data/ffhq_1bit_0.05_adm/output.yscores.txt \
	--title 'Stage 2: ROC for ADM' --color slateblue \
	--save stage2-roc-adm.pdf
python3 roc.py \
	--ytrue data/ffhq_1bit_0.05_ldm/img.ytrue.txt \
	--yscore data/ffhq_1bit_0.05_ldm/img.yscores.txt \
	--title 'Stage 2: ROC for LDM' --color slateblue \
	--save stage2-roc-ldm.pdf

