
echo 'EXPERIMENTS ISMB CV'


n_gcn_layers=3

python ChebModelWPathways_CV_hpsearch.py \
--out_dir 'out/chebmodelwpathways/final_CV/layers_'$n_gcn_layers'/raw' \
--device_idx 4 \
--norm False \
--num_layers  $n_gcn_layers \
--hidden_gcn 8 \
--epochs 100

