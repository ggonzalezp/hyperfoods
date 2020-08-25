
echo 'EXPERIMENTS ISMB CV'


n_gcn_layers=2

python MLPModel_CV_hpsearch.py \
--out_dir 'out/mlpmodel/final_CV/layers_'$n_gcn_layers'/raw' \
--device_idx 1 \
--norm False \
--num_layers  $n_gcn_layers \
--hidden_gcn 8 \
--epochs 100
