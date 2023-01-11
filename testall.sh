pathmodel="output/cls0.01_tanh"
pathout="samples/cls0.01_tanh"
pathdata="/data/yzeng22/imagenetval_sample1k_center256"
mkdir $pathout
files="
4000
8000
20000
40000
60000
80000
100000
110000
120000
130000
"

for file in $files
do
	python unet_recon_test.py --data_dir $pathdata --model_path $pathmodel/net_$file.pth  --log_dir $pathout/$file
	python unet_cls_test.py --data_dir $pathout/$file --model_path $pathmodel/net_cls_$file.pth  | grep TotalAcc | sed "s/^/file $file /" >> $pathout.acc.txt
	python -m pytorch_fid $pathdata $pathout/$file --device cuda:0  | grep FID | sed "s/^/file $file /" >> $pathout.fid.txt
	python psnr_opencv.py $pathout/$file $pathdata | grep TotalPSNR | sed "s/^/file $file /" >> $pathout.psnr.txt
done
