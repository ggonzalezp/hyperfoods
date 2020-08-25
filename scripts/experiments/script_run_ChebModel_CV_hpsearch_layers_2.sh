
echo 'EXPERIMENTS ISMB CV'


n_gcn_layers=2

python ChebModel_CV_hpsearch.py \
--out_dir 'out/chebmodel/final_CV/layers_'$n_gcn_layers'/raw' \
--device_idx 3 \
--norm False \
--num_layers  $n_gcn_layers \
--hidden_gcn 8 \
--epochs 100

