allfigs: fig-lambda fig-roc fig-vis-w
# gs -sDEVICE=pdfwrite -dPDFSETTINGS=/printer -q -o in out

fig-fwm-roc:
	python3 fwm-roc.py --save fwm-roc.pdf

fig-lambda:
	python3 lambda.py --save lambda.pdf

fig-vis-w:
	python3 visualize.py w --save vis-w.svg

fig-vis-g:
	python3 visualize.py g --save vis-g.svg

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
        --unseen_yscore data/ffhq_unseen_adm_320000/samples1k_pretrain.yscores.txt \
        --unseen_ytrue data/ffhq_unseen_adm_320000/samples1k_pretrain.ytrue.txt \
        --wang_yscore data/ffhq_cnndetection/ldm_result.yscores.txt \
        --wang_ytrue data/ffhq_cnndetection/ldm_result.ytrue.txt \
        --grag_yscore data/ffhq_ganimagedetection/ldm.yscores.txt \
        --grag_ytrue data/ffhq_ganimagedetection/ldm.ytrue.txt \
		--title 'Stage 2: ADM' --color slateblue \
		--save stage2-roc-adm.pdf
	python3 roc.py \
		--ytrue data/ffhq_1bit_0.05_ldm/img.ytrue.txt \
		--yscore data/ffhq_1bit_0.05_ldm/img.yscores.txt \
		--seen_ytrue data/ffhq_baseline_allseen_ldm_320000/samples1k_pretrain.ytrue.txt \
		--seen_yscore data/ffhq_baseline_allseen_ldm_320000/samples1k_pretrain.yscores.txt \
        --unseen_yscore data/ffhq_unseen_ldm_320000/samples1k_pretrain.yscores.txt \
        --unseen_ytrue data/ffhq_unseen_ldm_320000/samples1k_pretrain.ytrue.txt \
        --wang_yscore data/ffhq_cnndetection/adm_result.yscores.txt \
        --wang_ytrue data/ffhq_cnndetection/adm_result.ytrue.txt \
        --grag_yscore data/ffhq_ganimagedetection/adm.yscores.txt \
        --grag_ytrue data/ffhq_ganimagedetection/adm.ytrue.txt \
		--title 'Stage 2: LDM' --color slateblue \
		--save stage2-roc-ldm.pdf
	python3 roc.py \
		--yscore data/ffhq_1bit_0.05_stylegan/576000.pt.yscores.txt \
		--ytrue data/ffhq_1bit_0.05_stylegan/576000.pt.ytrue.txt \
		--seen_ytrue data/ffhq_baseline_allseen_stylegan_320000/samples1k_pretrain.ytrue.txt \
		--seen_yscore data/ffhq_baseline_allseen_stylegan_320000/samples1k_pretrain.yscores.txt \
        --unseen_yscore data/ffhq_unseen_stylegan_320000/samples1k_pretrain.yscores.txt \
        --unseen_ytrue data/ffhq_unseen_stylegan_320000/samples1k_pretrain.ytrue.txt \
        --grag_yscore data/ffhq_ganimagedetection/stylegan2.yscores.txt \
        --grag_ytrue data/ffhq_ganimagedetection/stylegan2.ytrue.txt \
        --wang_yscore data/ffhq_cnndetection/stylegan2.yscores.txt \
        --wang_ytrue data/ffhq_cnndetection/stylegan2.ytrue.txt \
		--title 'Stage 2: StyleGAN2' --color slateblue \
        --save stage2-roc-stylegan2.pdf
