# hyperfoods


Code repository for 'Predicting Anticancer Hyperfoods with Graph Convolutional Networks'


0. Install libraries

conda create --name hyperfoods python=3.6.9 pandas=0.25.1 matplotlib=3.1.2
conda activate hyperfoods
pip install pip==19.3
Adapt the following ones to your version of cuda
conda install pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.1 -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric==1.4.3
pip install hickle
pip install torchbnn

1. Preprocess data

python preprocess.py



2. Run the model
Model with Chebishev filter (cross-validation to optimize hyperparameters):

python ChebModel_CV_hpsearch.py --out_dir 'cheb_nlayers_2_hidden_8' --device_idx 3 --norm False --num_layers 2 --hidden_gcn 8 --epochs 100 --dataset_dir 'dataset/test' 


3. Compute attributions








