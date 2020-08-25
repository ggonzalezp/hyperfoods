
echo 'EXPERIMENTS ISMB CV'


n_gcn_layers=3

python MLPModel_CV_hpsearch.py \
--out_dir 'out/mlpmodel/final_CV/layers_'$n_gcn_layers'/norm' \
--device_idx 5 \
--norm True \
--num_layers  $n_gcn_layers \
--hidden_gcn 8 \
--epochs 100
