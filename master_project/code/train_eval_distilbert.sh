# DistilBERT ----------------------------------------------------------------------------

# create results folder and save the results
mkdir -p ./results/distilbert/strucpremise

python infotabs_trainer.py \
    --model "distilbert" \
	--mode "train" \
    --splits "train" "dev" "test_alpha1" "test_alpha2" "test_alpha3"\
    --data_tsv_dir "../temp/data/strucpremise/" \
    --out_dir "./results/distilbert/strucpremise/" \
	--epochs 10 \
	--batch_size 8 \
    --lr 2e-5 \
    --max_len 512 \
    2>&1 | tee -a ./results/distilbert/strucpremise_$(date +%Y%m%d_%H%M%S).txt
