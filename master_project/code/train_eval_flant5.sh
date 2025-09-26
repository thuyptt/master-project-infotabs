# Flan-T5-small ----------------------------------------------------------------------------

# create results folder and save the results
mkdir -p ./results/flan-t5-small/strucpremise

python pipeline_flant5.py \
    --model "flan-t5-small" \
	--mode "train" \
    --splits "train" "dev" "test_alpha1" "test_alpha2" "test_alpha3" \
    --data_tsv_dir "../temp/data/strucpremise/" \
    --out_dir "./results/flan-t5-small/strucpremise/" \
	--epochs 10 \
	--batch_size 24 \
    --lr 2e-5 \
    --max_len 512 \
    2>&1 | tee -a ./results/flan-t5-small/strucpremise_$(date +%Y%m%d_%H%M%S).txt

