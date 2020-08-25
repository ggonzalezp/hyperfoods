
echo 'EXPERIMENTS ISMB CV'


n_gcn_layers=1

python SageModel_CV_hpsearch.py \
--out_dir 'out/sagemodel/final_CV/layers_'$n_gcn_layers'/raw' \
--device_idx 4 \
--norm False \
--num_layers  $n_gcn_layers \
--hidden_gcn 8 \
--epochs 100
