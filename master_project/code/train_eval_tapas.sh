# TAPAS-small ----------------------------------------------------------------------------

# create results folder and save the results
mkdir -p ./results/tapas-small

python pipeline_tapas.py \
    --model "tapas-small" \
	--mode "train" \
    --splits "train" "dev" "test_alpha1" "test_alpha2" "test_alpha3" \
    --data_tsv_dir "../data/maindata/" \
    --tables_dir "../data/tables/" \
    --out_dir "./results/tapas-small/" \
	--epochs 10 \
	--batch_size 70 \
    --lr 2e-5 \
    --max_len 512 \
    2>&1 | tee -a ./results/tapas-small/tapas_$(date +%Y%m%d_%H%M%S).txt

