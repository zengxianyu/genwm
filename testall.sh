pathmodel="output/q_start100_cls1.0_tanh"
pathout="samples/q_start100_cls1.0_tanh"
pathdata="/data/FFHQ/images256x256_sample1k"
mkdir $pathout
#files="
#4000
#8000
#20000
#40000
#60000
#80000
#"
#files="
#100000
#120000
#140000
#160000
#180000
#200000
#"
files="
200000
"

for file in $files
do
	python unet_recon_test.py --data_dir $pathdata --model_path $pathmodel/net_$file.pth  --log_dir $pathout/$file
	python unet_cls_test.py --gt_dir $pathdata --data_dir $pathout/$file --model_path $pathmodel/net_cls_$file.pth  | grep TotalAcc | sed "s/^/file $file /" >> $pathout.acc.txt
	python -m pytorch_fid $pathdata $pathout/$file --device cuda:0  | grep FID | sed "s/^/file $file /" >> $pathout.fid.txt
	python psnr_opencv.py $pathout/$file $pathdata | grep TotalPSNR | sed "s/^/file $file /" >> $pathout.psnr.txt
done
