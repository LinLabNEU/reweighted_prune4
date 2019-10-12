export CUDA_VISIBLE_DEVICES=0,1

python -u main.py --arch mobilenetv2 \
                                           --quant_method linear\
                                           --param_bits 8\
                                           --fwd_bits 8\
					   --depth 17 \
					   --batch-size 64 \
					   --no-tricks \
					   --masked-retrain \
					   --sparsity-type threshold \
					   --epoch 300 \
					   --optmzr adam \
					   --lr 0.001 \
					   --lr-scheduler cosine \
					   --warmup \
					   --warmup-epochs 5 \
					   --mixup \
					   --alpha 0.3 \
					   --smooth \
					   --smooth-eps 0.1 


