
echo 'EXPERIMENTS ISMB CV'


n_gcn_layers=1

python SageModelWPathways_CV_hpsearch.py \
--out_dir 'out/sagemodelwpathways/final_CV/layers_'$n_gcn_layers'/norm' \
--device_idx 3 \
--norm True \
--num_layers  $n_gcn_layers \
--hidden_gcn 8 \
--epochs 100
