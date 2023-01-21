pathmodel="ffhq/q_start100_cls0.1_tanh_patch_v6"
pathout="samples_patch/q_start100_cls1.0_tanh"
pathdata="/data/FFHQ/images256x256_sample1k"
mkdir $pathout
files="
24000
26000
28000
30000
32000
34000
36000
38000
40000
42000
44000
46000
48000
50000
52000
54000
56000
58000
60000
62000
64000
66000
68000
70000
72000
74000
76000
"
#files="
#100000
#120000
#140000
#160000
#180000
#200000
#"
#files="
#200000
#"

for file in $files
do
	python unet_recon_test.py --data_dir $pathdata --model_path $pathmodel/net_$file.pth  --log_dir $pathout/$file
	python unet_cls_test.py --gt_dir $pathdata --data_dir $pathout/$file --model_path $pathmodel/net_cls_$file.pth  | grep TotalAcc | sed "s/^/file $file /" >> $pathout.acc.txt
	python -m pytorch_fid $pathdata $pathout/$file --device cuda:0  | grep FID | sed "s/^/file $file /" >> $pathout.fid.txt
	python psnr_opencv.py $pathout/$file $pathdata | grep TotalPSNR | sed "s/^/file $file /" >> $pathout.psnr.txt
done
