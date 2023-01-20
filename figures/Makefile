allfigs: fig-lambda fig-roc fig-vis-w

fig-lambda:
	python3 lambda.py --save lambda.pdf

fig-vis-w:
	python3 visualize.py w --save vis-w.svg
	# manually edit and export as pdf

fig-roc:
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
		--seen_yscore data/ffhq_baseline_allseen_adm_320000/samples1k_pretrain.yscores.txt \
		--seen_ytrue data/ffhq_baseline_allseen_adm_320000/samples1k_pretrain.ytrue.txt \
		--title 'Stage 2: ADM' --color slateblue \
		--save stage2-roc-adm.pdf
	python3 roc.py \
		--ytrue data/ffhq_1bit_0.05_ldm/img.ytrue.txt \
		--yscore data/ffhq_1bit_0.05_ldm/img.yscores.txt \
		--seen_ytrue data/ffhq_baseline_allseen_ldm_320000/samples1k_pretrain.ytrue.txt \
		--seen_yscore data/ffhq_baseline_allseen_ldm_320000/samples1k_pretrain.yscores.txt \
		--title 'Stage 2: LDM' --color slateblue \
		--save stage2-roc-ldm.pdf
	# stylegan2 curve still missing


