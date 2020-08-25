
echo 'EXPERIMENTS ISMB CV'


n_gcn_layers=1

python MLPModel_CV_hpsearch.py \
--out_dir 'out/mlpmodel/final_CV/layers_1/raw' \
--device_idx 0 \
--norm False \
--num_layers  $n_gcn_layers \
--hidden_gcn 8 \
--epochs 100
