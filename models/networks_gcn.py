import torch
from torch_geometric.nn import TopKPooling, global_mean_pool, global_add_pool, global_max_pool, Set2Set
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch.nn import Linear, BatchNorm1d
from models.conv.gcn_conv import GCNConv
from models.conv.cheb_conv import ChebConv
from models.conv.sage_conv import SAGEConv
from models.conv.gat_conv import GATConv
from models.conv.gmm_conv import GMMConv
import torchbnn as bnn
import numpy as np
from torch_geometric.utils import to_dense_adj
###########
## this networks use a custom implementation of GCN (by SW) to use only one edge_index
## the DataLoader is also a custom one -- we don't need the edge_index now for the samples or the forward function
## edge_index is passed when constructing the model
## DataLoader provides samples with dimensions [nbatch, ngenes, in_channels]
## This one is directly compatible with the input dimension of data required by captum





########Base models - GCN
#FC
class GCNModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, n_genes = 15135, mode='cat', bayesian = False, batchnorm = False, do_layers = 1, edge_weights = None):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = GCNConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.do_layers =do_layers
        self.mode = mode
        self.edge_weights = edge_weights


        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_gcn, hidden_gcn))

        if mode == 'cat':
                self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1
        else:
                self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1


        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_genes, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_genes, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)

        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))

    def graph_embedding(self, x, batch):

        ##Computes graph embedding: GCN(s) + FC(1)

        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)

        #####   GCN
        #First GCN layer
        x = F.relu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index, edge_weights)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index, edge_weights))


        #######     FC(1)
        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)

        return x




    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]

        ##Graph embedding
        x = self.graph_embedding(x, batch)


        ##Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)


#Pathways + FC
class GCNModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat', bayesian = False, batchnorm = False, n_genes = 15135, do_layers = 1, edge_weights = None):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers
        self.edge_weights = edge_weights

        #GCNs
        self.conv1 = GCNConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_gcn, hidden_gcn))


        #FC(1)
        if mode == 'cat':
            self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)              #FC layer to reduce dim of pathway features to 1
        else:
            self.fc = torch.nn.Linear(hidden_gcn, 1)              #FC layer to reduce dim of pathway features to 1


        #MLP
        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_cmt, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_cmt, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)


        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))

    def graph_embedding(self, x, batch):
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)


        ######  GCNs

        x = F.relu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

        #Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]


            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index, edge_weights)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index, edge_weights))


        #Pathway pooling
        x = x.view(bs, -1, x.size(-1))  # [b_size, n_nodes, features]
        x = x.transpose(0, 1)
        x = scatter_mean(x[self.row], self.col, dim=0)
        x = x.transpose(0, 1).contiguous().view(-1, self.fc.weight.size(1))

        #FC to aggregate pathway features
        x = self.fc(x).view(-1, self.lin1.weight.size(1))

        return x




    def forward(self,  x,  batch):
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        #Graph embedding
        x = self.graph_embedding(x, batch)


        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training) #dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training) #dropout
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)







########ChebNet models
#FC
class ChebModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, n_genes=15135, mode='cat',
                 bayesian=False, batchnorm=False, k_hops=2, do_layers=1, edge_weights = None):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = ChebConv(n_features, hidden_gcn, k_hops)
        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.do_layers = do_layers
        self.edge_weights = edge_weights

        self.mode = mode

        for i in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_gcn, hidden_gcn, k_hops))

        if mode == 'cat':
            self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1
        else:
            self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1

        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_genes, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_genes, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)

        # BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))

    def graph_embedding(self, x, batch):

        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)

        # First GCN layer
        x = F.relu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index, edge_weights)).view(bs * num_nodes, -1)).view(bs, num_nodes,
                                                                                                       -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index, edge_weights))

        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)

        return x


    def forward(self, x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]

        #Graph embedding
        x = self.graph_embedding(x,batch)


        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)



#Pathways + FC
class ChebModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat',
                 bayesian=False, batchnorm=False, n_genes=15135, k_hops=2, do_layers=1, edge_weights = None):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers
        self.edge_weights = edge_weights

        # GCNs
        self.conv1 = ChebConv(n_features, hidden_gcn, k_hops)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_gcn, hidden_gcn, k_hops))

        # FC(1)
        if mode == 'cat':
            self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1
        else:
            self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1

        # MLP
        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_cmt, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_cmt, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)

        # BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))

    def graph_embedding(self, x, batch):

        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)

        ######  GCNs

        x = F.relu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

        # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index, edge_weights)).view(bs * num_nodes, -1)).view(bs, num_nodes,
                                                                                                       -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index, edge_weights))

        # Pathway pooling
        x = x.view(bs, -1, x.size(-1))  # [b_size, n_nodes, features]
        x = x.transpose(0, 1)
        x = scatter_mean(x[self.row], self.col, dim=0)
        x = x.transpose(0, 1).contiguous().view(-1, self.fc.weight.size(1))

        # FC to aggregate pathway features
        x = self.fc(x).view(-1, self.lin1.weight.size(1))

        return x

    def forward(self, x, batch):
        # x input dim [nbatch, nnodes, nfeatures]

        #Graph embedding
        x = self.graph_embedding(x, batch)

        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)  # dropout
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)





########GraphSage models

#FC
class SageModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, n_genes = 15135, mode='cat', bayesian = False, batchnorm = False, do_layers = 1, edge_weights = None):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = SAGEConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.do_layers =do_layers
        self.edge_weights = edge_weights

        self.mode = mode

        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_gcn, hidden_gcn))

        if mode == 'cat':
                self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1
        else:
                self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1


        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_genes, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_genes, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)

        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))

    def graph_embedding(self, x, batch):
        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)


        #First GCN layer
        x = F.relu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index, edge_weights)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index, edge_weights))




        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)
        return x




    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]

        #Graph embedding
        x = self.graph_embedding(x, batch)

        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)



#Pathways + FC
class SageModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat', bayesian = False, batchnorm = False, n_genes = 15135, do_layers = 1, edge_weights = None):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers
        self.edge_weights = edge_weights

        #GCNs
        self.conv1 = SAGEConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_gcn, hidden_gcn))


        #FC(1)
        if mode == 'cat':
            self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)              #FC layer to reduce dim of pathway features to 1
        else:
            self.fc = torch.nn.Linear(hidden_gcn, 1)              #FC layer to reduce dim of pathway features to 1


        #MLP
        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_cmt, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_cmt, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)


        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))

    def graph_embedding(self, x, batch):
        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)

        ######  GCNs

        x = F.relu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

        #Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]


            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index, edge_weights)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index, edge_weights))


        #Pathway pooling
        x = x.view(bs, -1, x.size(-1))  # [b_size, n_nodes, features]
        x = x.transpose(0, 1)
        x = scatter_mean(x[self.row], self.col, dim=0)
        x = x.transpose(0, 1).contiguous().view(-1, self.fc.weight.size(1))

        #FC to aggregate pathway features
        x = self.fc(x).view(-1, self.lin1.weight.size(1))
        return x




    def forward(self,  x,  batch):
        # x input dim [nbatch, nnodes, nfeatures]

        #Graph embedding
        x = self.graph_embedding(x, batch)


        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training) #dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training) #dropout
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)






########Base models - MLP
#FC
class MLPModel(torch.nn.Module):

    def __init__(self, n_features, n_classes, num_layers, hidden, hidden_fc, n_genes = 15135, mode = 'cat', batchnorm = False, do_layers = 1):
        super().__init__()
        self.batchnorm = batchnorm
        self.mode = mode
        self.do_layers = do_layers

        #MLP
        self.mlp1 = Linear(n_features, hidden)
        self.mlps = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.mlps.append(Linear(hidden, hidden))

        #FC(1)
        if mode == 'cat':
                self.fc = torch.nn.Linear(num_layers * hidden, 1)  # FC layer to reduce dim of pathway features to 1
        else:
                self.fc = torch.nn.Linear(hidden, 1)  # FC layer to reduce dim of pathway features to 1


        #Prediction layers
        self.lin1 = Linear(n_genes, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)


        #BatchNorm
        if batchnorm:
            self.bnmlps = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnmlps.append(BatchNorm1d(hidden))


    def mlp(self, x, batch):
        bs, num_nodes = x.size(0), x.size(1)

        #First MLP layer
        x = F.relu(self.mlp1(x))
        if self.batchnorm:
            x = self.bnmlps[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)


            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for layer in self.mlps:
                x = F.relu(layer(x))

                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)


        elif self.mode == 'base':
            for layer in self.mlps:
                x = F.relu(layer(x))
                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for layer in self.mlps:
                if self.batchnorm:
                    x = x + self.bnmlps[i](F.relu(layer(x)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(layer(x))


        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)
        return x



    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]

        x = self.mlp(x, batch)

        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)


#Pathways + FC
class MLPModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden, hidden_fc, pathway, n_cmt, mode='cat', batchnorm = False, n_genes = 15135, do_layers = 1):
        super().__init__()
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers

        #GCNs
        self.mlp1 = Linear(n_features, hidden)
        self.mlps = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.mlps.append(Linear(hidden, hidden))


        #FC(1)
        if mode == 'cat':
            self.fc = torch.nn.Linear(num_layers * hidden, 1)              #FC layer to reduce dim of pathway features to 1
        else:
            self.fc = torch.nn.Linear(hidden, 1)              #FC layer to reduce dim of pathway features to 1


        #MLP
        self.lin1 = Linear(n_cmt, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)


        #BatchNorm
        if batchnorm:
            self.bnmlps = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnmlps.append(BatchNorm1d(hidden))



    def mlp(self, x, batch):
        # x input dim [nbatch, nnodes, nfeatures]
        bs, num_nodes = x.size(0), x.size(1)


        #First MLP layer
        x = F.relu(self.mlp1(x))
        if self.batchnorm:
            x = self.bnmlps[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)


            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for layer in self.mlps:
                x = F.relu(layer(x))

                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)


        elif self.mode == 'base':
            for layer in self.mlps:
                x = F.relu(layer(x))
                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for layer in self.mlps:
                if self.batchnorm:
                    x = x + self.bnmlps[i](F.relu(layer(x)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(layer(x))


        #Pathway pooling
        x = x.view(bs, -1, x.size(-1))  # [b_size, n_nodes, features]
        x = x.transpose(0, 1)
        x = scatter_mean(x[self.row], self.col, dim=0)
        x = x.transpose(0, 1).contiguous().view(-1, self.fc.weight.size(1))

        #FC to aggregate pathway features
        x = self.fc(x).view(-1, self.lin1.weight.size(1))

        return x




    def forward(self,  x,  batch):
        # x input dim [nbatch, nnodes, nfeatures]
        bs, num_nodes = x.size(0), x.size(1)

        x = self.mlp(x, batch)

        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)



########Base models - FC MODEL - FC layer for each of the genes
#FC
## Model:
class FCModel(torch.nn.Module):

    def __init__(self, n_features, n_classes, num_layers, hidden_fc, n_genes = 15135, mode = 'cat', batchnorm = False, do_layers = 1):
        super().__init__()
        self.batchnorm = batchnorm
        self.mode = mode
        self.do_layers = do_layers

        #MLP                --- I will transpose my input matrix (Ngenes x 1) to apply FC on the whole set of genes XW ([1 x ngenes] x [ngenes x ngenes])
        self.mlp1 = Linear(n_genes, n_genes)
        self.mlps = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.mlps.append(Linear(n_genes, n_genes))

        #Prediction layers
        self.lin1 = Linear(n_genes, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)


        #BatchNorm
        if batchnorm:
            self.bnmlps = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnmlps.append(BatchNorm1d(n_features))


    def mlp(self, x, batch):
        bs, num_nodes = x.size(0), x.size(1)

        #First MLP layer
        x = F.relu(self.mlp1(x.transpose(1,2)).transpose(1,2))
        if self.batchnorm:
            x = self.bnmlps[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)


            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for layer in self.mlps:
                x = F.relu(layer(x.transpose(1,2)).transpose(1,2))

                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)


        elif self.mode == 'base':
            for layer in self.mlps:
                x = F.relu(layer(x.transpose(1,2)).transpose(1,2))
                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for layer in self.mlps:
                if self.batchnorm:
                    x = x + self.bnmlps[i](F.relu(layer(x.transpose(1,2)).transpose(1,2)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(layer(x.transpose(1,2)).transpose(1,2))

                    ###shape [ nbatch, ngenes]
        return x.view(bs, num_nodes)



    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]

        x = self.mlp(x, batch)

        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

########Base models - FC MODEL - FC layer for each of the genes ONLY NEIGHBORS

class FCNModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_fc, edge_index,device, n_genes = 15135, mode='cat', bayesian = False, batchnorm = False, do_layers = 1, edge_weights = None, init_matrix = None):
        super().__init__()
        self.adj_matrix = to_dense_adj(edge_index).view(n_genes, n_genes) + torch.eye(n_genes).to(device)



        self.batchnorm = batchnorm
        self.do_layers =do_layers
        self.mode = mode
        self.edge_weights = edge_weights

        #MLP                --- I will transpose my input matrix (Ngenes x 1) to apply FC on the whole set of genes XW ([1 x ngenes] x [ngenes x ngenes])
        self.mlp1 = Linear(n_genes, n_genes)
        if init_matrix != None:
            self.mlp1.weight.data = init_matrix
        self.mlps = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            layer = Linear(n_genes, n_genes)
            if init_matrix != None:
                layer.weight.data = init_matrix
            self.mlps.append(layer)




        self.lin1 = Linear(n_genes, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)

        #BatchNorm
        if batchnorm:
            self.bnmlps = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnmlps.append(BatchNorm1d(n_features))




    def mlp(self, x, batch):

        bs, num_nodes = x.size(0), x.size(1)

        #First MLP layer
        self.mlp1.weight = torch.nn.Parameter(self.mlp1.weight * self.adj_matrix, requires_grad = True)             #masking weights
        x = F.relu(self.mlp1(x.transpose(1,2)).transpose(1,2))
        if self.batchnorm:
            x = self.bnmlps[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)


            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for layer in self.mlps:
                layer.weight = torch.nn.Parameter(layer.weight * self.adj_matrix, requires_grad = True)
                x = F.relu(layer(x.transpose(1,2)).transpose(1,2))

                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)


        elif self.mode == 'base':
            for layer in self.mlps:
                x = F.relu(layer(x.transpose(1,2)).transpose(1,2))
                if self.batchnorm:
                    x = self.bnmlps[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for layer in self.mlps:
                if self.batchnorm:
                    x = x + self.bnmlps[i](F.relu(layer(x.transpose(1,2)).transpose(1,2)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(layer(x.transpose(1,2)).transpose(1,2))

                    ###shape [ nbatch, ngenes]
        return x.view(bs, num_nodes)




    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]


        ##Graph embedding
        x = self.mlp(x, batch)


        ##Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)








########Base models - GCN
#FC
class GCNEModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, n_genes = 15135, mode='cat', batchnorm = False, do_layers = 1, edge_weights = None, pembedding_dim = 32, device = None):
        super().__init__()
        self.edge_index = edge_index

        self.batchnorm = batchnorm
        self.do_layers =do_layers
        self.mode = mode
        self.edge_weights = edge_weights

        ######Positional embedding
        #Onehot node identifiers (for positional embedding)
        self.nodes_onehot = torch.eye(n_genes).to(device)
        self.pelayer = torch.nn.Linear(n_genes, pembedding_dim)




        #######Graph embedding

        self.conv1 = GCNConv(n_features + pembedding_dim, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_gcn, hidden_gcn))

        if mode == 'cat':
                self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1
        else:
                self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1



        self.lin1 = Linear(n_genes, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)

        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))

    def graph_embedding(self, x, batch):

        ##Computes graph embedding: GCN(s) + FC(1)

        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)

        #####   GCN
        #First GCN layer
        x = F.relu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weights))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index, edge_weights)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index, edge_weights))


        #######     FC(1)
        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)

        return x




    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]
        bs = x.size(0)

        #Positional graph_embedding
        pe = self.pelayer(self.nodes_onehot).view(1, self.nodes_onehot.size(0), -1)

        x = torch.cat((x, torch.cat([pe] * bs)), 2)     #concat positional embedding + node features

        ##Graph embedding
        x = self.graph_embedding(x, batch)


        ##Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)


class GCNGraphLetModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, nodes_graphlets, graphlet_mode = 'input', pembedding_dim = 8, n_genes = 15135, mode='cat', batchnorm = False, do_layers = 1, edge_weights = None):
        super().__init__()

        self.edge_index = edge_index

        self.batchnorm = batchnorm
        self.do_layers =do_layers
        self.mode = mode
        self.edge_weights = edge_weights
        self.graphlet_mode = graphlet_mode

        ######Positional embedding
        #Onehot node identifiers (for positional embedding)
        self.nodes_graphlets = nodes_graphlets
        self.pelayer = torch.nn.Linear(nodes_graphlets.size(1), pembedding_dim) #embedding layer to reduce dim of positional embedding


        #######Graph embedding


        self.convs = torch.nn.ModuleList()




        if graphlet_mode == 'input':        ###graphlets are concatenated only in the input
            self.conv1 = GCNConv(n_features + nodes_graphlets.size(1), hidden_gcn)
            for i in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_gcn, hidden_gcn))
        elif graphlet_mode == 'all':        ###graphlets are concatenated in every layer
            self.conv1 = GCNConv(n_features + nodes_graphlets.size(1), hidden_gcn)
            for i in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_gcn + nodes_graphlets.size(1), hidden_gcn))

        elif graphlet_mode == 'input_emb':      ##graphlets are passed through an embedding layer and then concatenated to input featires
            self.conv1 = GCNConv(n_features + pembedding_dim, hidden_gcn)
            for i in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_gcn, hidden_gcn))
        elif graphlet_mode == 'out':        ###graphlet only after all the conv layers (before the fc layer)
            self.conv1 = GCNConv(n_features, hidden_gcn)

            for i in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_gcn, hidden_gcn))






        if mode == 'cat':
            if graphlet_mode == 'out':
                self.fc0 = torch.nn.Linear(num_layers * hidden_gcn + nodes_graphlets.size(1), num_layers* hidden_gcn )

            self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1
        else:
            if graphlet_mode == 'out':
                self.fc0 = torch.nn.Linear(hidden_gcn + nodes_graphlets.size(1), hidden_gcn )
            self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1



        self.lin1 = Linear(n_genes, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)

        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))



    def graph_embedding(self, x, batch):

        ##Computes graph embedding: GCN(s) + FC(1)

        edge_index = self.edge_index
        edge_weights = self.edge_weights
        bs, num_nodes = x.size(0), x.size(1)

        pe = self.nodes_graphlets.view(1, self.nodes_graphlets.size(0), -1)



        #####   GCN
        #First GCN layer

        x = F.elu(self.conv1(x, edge_index, edge_weights))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                if self.graphlet_mode == 'all':
                    x = torch.cat((x, torch.cat([pe] * bs)), 2)
                x = F.elu(conv(x, edge_index, edge_weights))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)




        #######     FC(1)
        # FC to aggregate pathway features
        if self.graphlet_mode == 'out':
            x = torch.cat((x, torch.cat([pe] * bs)), 2)
            x = self.fc0(x)

        x = self.fc(x).view(bs, num_nodes)

        return x




    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]
        bs = x.size(0)

        #Positional graph_embedding
        pe = self.nodes_graphlets.view(1, self.nodes_graphlets.size(0), -1)

        if self.graphlet_mode == 'input_emb':
            pe = self.pelayer(pe)

        if self.graphlet_mode != 'out':
            x = torch.cat((x, torch.cat([pe] * bs)), 2)     #concat positional embedding (node graphlets) + node features

        ##Graph embedding
        x = self.graph_embedding(x, batch)


        ##Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.elu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

class GMMModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, pseudo, n_genes=15135, mode='cat',
                    batchnorm=False, do_layers=1, kernel_size = 2):
        super().__init__()
        self.edge_index = edge_index

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.do_layers = do_layers
        self.pseudo = pseudo
        pseudo_dim = pseudo.size(1)
        self.mode = mode


        self.conv1 = GMMConv(n_features, hidden_gcn, dim = pseudo_dim, kernel_size = kernel_size)
        for i in range(num_layers - 1):
            self.convs.append(GMMConv(hidden_gcn, hidden_gcn, dim = pseudo_dim, kernel_size = kernel_size))

        if mode == 'cat':
            self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1
        else:
            self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1


        self.lin1 = Linear(n_genes, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)

        # BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn))






    def graph_embedding(self, x, batch):

        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        pseudo = torch.cat([self.pseudo] * bs).view(bs, self.pseudo.size(0), self.pseudo.size(1))

        # First GCN layer
        x = F.elu(self.conv1(x, edge_index, pseudo))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.elu(conv(x, edge_index, pseudo))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.elu(conv(x, edge_index, pseudo))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.elu(conv(x, edge_index, pseudo)).view(bs * num_nodes, -1)).view(bs, num_nodes,
                                                                                                       -1)
                    i += 1
                else:
                    x = x + F.elu(conv(x, edge_index, pseudo))

        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)

        return x


    def forward(self, x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]

        #Graph embedding


        x = self.graph_embedding(x,batch)


        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.elu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)
















######Not used : GAT too much memory still

########    GAT models
#FC
class GATModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, n_genes = 15135, mode='cat', bayesian = False, batchnorm = False, do_layers = 1):
        super().__init__()
        self.edge_index = edge_index
        self.batchnorm = batchnorm
        self.do_layers =do_layers
        n_heads = 1



        self.conv1 = GATConv(n_features, hidden_gcn, heads = n_heads, dropout = 0.5)
        self.convs = torch.nn.ModuleList()


        self.mode = mode

        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden_gcn * n_heads , hidden_gcn, heads = n_heads, dropout = 0.5))

        if mode == 'cat':
                self.fc = torch.nn.Linear(num_layers * hidden_gcn * n_heads, 1)  # FC layer to reduce dim of pathway features to 1
        else:
                self.fc = torch.nn.Linear(hidden_gcn * h_heads, 1)  # FC layer to reduce dim of pathway features to 1


        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_genes, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_genes, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)

        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn * n_heads))

    def graph_embedding(self, x, batch):

        ##Computes graph embedding: GCN(s) + FC(1)

        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        #####   GCN
        #First GCN layer
        x = F.relu(self.conv1(x, edge_index))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

            # Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index))


        #######     FC(1)
        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)

        return x




    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]




        ##Graph embedding
        x = self.graph_embedding(x, batch)


        ##Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)



#Pathways + FC
class GATModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat', bayesian = False, batchnorm = False, n_genes = 15135, do_layers = 1):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers

        n_heads = 8

        #GCNs
        self.conv1 = GATConv(n_features, hidden_gcn, heads = n_heads, dropout = 0.5)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden_gcn * n_heads, hidden_gcn, heads = n_heads, dropout = 0.5))


        #FC(1)
        if mode == 'cat':
            self.fc = torch.nn.Linear(num_layers * hidden_gcn * n_heads, 1)              #FC layer to reduce dim of pathway features to 1
        else:
            self.fc = torch.nn.Linear(hidden_gcn * n_heads, 1)              #FC layer to reduce dim of pathway features to 1


        #MLP
        if bayesian:
            self.lin1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=n_cmt, out_features=hidden_fc)
            self.lin2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_fc, out_features=n_classes)

        else:
            self.lin1 = Linear(n_cmt, hidden_fc)
            self.lin2 = Linear(hidden_fc, n_classes)


        #BatchNorm
        if batchnorm:
            self.bnconvs = torch.nn.ModuleList()
            for i in range(num_layers):
                self.bnconvs.append(BatchNorm1d(hidden_gcn * n_heads))

    def graph_embedding(self, x, batch):
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)


        ######  GCNs

        x = F.relu(self.conv1(x, edge_index))
        if self.batchnorm:
            x = self.bnconvs[0](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)

        #Different feature aggregation
        i = 1
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

                xs += [x]


            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                if self.batchnorm:
                    x = self.bnconvs[i](x.view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1

        elif self.mode == 'sum':
            for conv in self.convs:
                if self.batchnorm:
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index)).view(bs * num_nodes, -1)).view(bs, num_nodes, -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index))


        #Pathway pooling
        x = x.view(bs, -1, x.size(-1))  # [b_size, n_nodes, features]
        x = x.transpose(0, 1)
        x = scatter_mean(x[self.row], self.col, dim=0)
        x = x.transpose(0, 1).contiguous().view(-1, self.fc.weight.size(1))

        #FC to aggregate pathway features
        x = self.fc(x).view(-1, self.lin1.weight.size(1))

        return x




    def forward(self,  x,  batch):
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        #Graph embedding
        x = self.graph_embedding(x, batch)


        #Prediction layers
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training) #dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training) #dropout
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)











#####Not used for now in CV

#TopK + FC
class GCNModelWTopK(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index,n_genes = 15135, mode='cat'):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = GCNConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        self.mode = mode

        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_gcn, hidden_gcn))



        if mode == 'cat':
            self.topkpooling = TopKPooling(num_layers * hidden_gcn)
            self.fc = torch.nn.Linear(num_layers * hidden_gcn, 1)  # FC layer to reduce dim of gene features to 1
        else:
            self.topkpooling = TopKPooling(hidden_gcn)
            self.fc = torch.nn.Linear(hidden_gcn, 1)  # FC layer to reduce dim of pathway features to 1

        n_nodes_after_topk = int(np.ceil(0.5 * n_genes))
        self.lin1 = Linear(n_nodes_after_topk, hidden_fc)
        self.lin2 = Linear(hidden_fc, n_classes)



    def forward(self,  x, batch):

        #x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        #First GCN layer
        x = F.relu(self.conv1(x, edge_index))


        #Different feature aggregation
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                xs += [x]
            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

        elif self.mode == 'sum':
            for conv in self.convs:
                x = x + F.relu(conv(x, edge_index))

        # import pdb; pdb.set_trace();

        # TopK Graph pooling
        xs = []
        for i in range(x.size(0)):
            xi, edge_index, edge_attr, batch_i, perm, score = self.topkpooling(x[i, :, :], self.edge_index)
            xs.append(xi)
        x = torch.stack(xs, 0)


        #FC to reduce dim of fealtures
        x = self.fc(x).view(bs, -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)








## With mean/max/add/set2set (global) pooling instead of FC

#mean/max/add/set2set
class GCNModel_pooled(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, mode='cat', graph_agg_mode = 'mean'):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = GCNConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        self.graph_agg_mode = graph_agg_mode
        self.mode = mode

        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_gcn, hidden_gcn))


        if mode == 'cat':
            if graph_agg_mode == 'set2set':
                self.lin1 = Linear(2*num_layers * hidden_gcn, hidden_fc)
                self.set2setpooling = Set2Set(num_layers * hidden_gcn, 2)
            else:
                self.lin1 = Linear(num_layers * hidden_gcn, hidden_fc)


        else:
            if graph_agg_mode == 'set2set':
                self.lin1 = Linear(hidden_gcn * 2, hidden_fc)
                self.set2setpooling = Set2Set(hidden_gcn,2)
            else:
                self.lin1 = Linear(hidden_gcn, hidden_fc)


        self.lin2 = Linear(hidden_fc, n_classes)



    def forward(self,  x, batch):

        #x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        #First GCN layer
        x = F.relu(self.conv1(x, edge_index))


        #Different feature aggregation
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

        elif self.mode == 'sum':
            for conv in self.convs:
                x = x + F.relu(conv(x, edge_index))

        # import pdb; pdb.set_trace();

        # Graph pooling
        if self.graph_agg_mode == 'mean':
            x = global_mean_pool(x.view(num_nodes * bs, -1), batch)

        elif self.graph_agg_mode == 'add':
            x = global_add_pool(x.view(num_nodes * bs, -1), batch)

        elif self.graph_agg_mode == 'max':
            x = global_max_pool(x.view(num_nodes * bs, -1), batch)

        elif self.graph_agg_mode == 'set2set':
            x = self.set2setpooling(x.view(num_nodes * bs, -1), batch)





        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)




#topK + mean/max/add/set2set
class GCNModelWTopK_pooled(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, mode='cat', graph_agg_mode = 'mean'):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = GCNConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        self.graph_agg_mode = graph_agg_mode
        self.mode = mode

        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_gcn, hidden_gcn))



        if mode == 'cat':
            self.topkpooling = TopKPooling(num_layers * hidden_gcn)
            if graph_agg_mode == 'set2set':
                self.lin1 = Linear(2*num_layers * hidden_gcn, hidden_fc)
                self.set2setpooling = Set2Set(num_layers * hidden_gcn, 2)
            else:
                self.lin1 = Linear(num_layers * hidden_gcn, hidden_fc)

        else:
            if graph_agg_mode == 'set2set':
                self.lin1 = Linear(hidden_gcn * 2, hidden_fc)
                self.set2setpooling = Set2Set(hidden_gcn,2)
            else:
                self.lin1 = Linear(hidden_gcn, hidden_fc)
            self.topkpooling = TopKPooling(hidden_gcn)



        self.lin2 = Linear(hidden_fc, n_classes)



    def forward(self,  x, batch):

        #x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        #First GCN layer
        x = F.relu(self.conv1(x, edge_index))


        #Different feature aggregation
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                xs += [x]
            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

        elif self.mode == 'sum':
            for conv in self.convs:
                x = x + F.relu(conv(x, edge_index))

        # import pdb; pdb.set_trace();
        # TopK Graph pooling
        xs = []
        for i in range(x.size(0)):
            xi, edge_index, edge_attr, batch_i, perm, score = self.topkpooling(x[i, :, :], self.edge_index)
            #global pooling
            if self.graph_agg_mode == 'mean':
                xi = global_mean_pool(xi, batch_i)
            elif self.graph_agg_mode == 'add':
                xi = global_add_pool(xi, batch_i)
            elif self.graph_agg_mode == 'max':
                xi = global_max_pool(xi, batch_i)
            elif self.graph_agg_mode == 'set2set':
                xi = self.set2setpooling(xi, batch_i)
            xs.append(xi)
        x = torch.stack(xs, 0).view(bs, -1)





        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)



#Pathways + mean/max/add/set2set
class GCNModelWPathways_pooled(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat', graph_agg_mode = 'mean'):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.graph_agg_mode = graph_agg_mode

        self.conv1 = GCNConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_gcn, hidden_gcn))


        if mode == 'cat':
            if graph_agg_mode == 'set2set':
                self.lin1 = Linear(2 * num_layers * hidden_gcn, hidden_fc)
                self.set2setpooling = Set2Set(num_layers * hidden_gcn, 2)
            else:
                self.lin1 = Linear(num_layers * hidden_gcn, hidden_fc)
        else:
            if graph_agg_mode == 'set2set':
                self.lin1 = Linear(hidden_gcn * 2, hidden_fc)
                self.set2setpooling = Set2Set(hidden_gcn, 2)
            else:
                self.lin1 = Linear(hidden_gcn, hidden_fc)



        self.lin2 = Linear(hidden_fc, n_classes)



    def forward(self,  x,  batch):

        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)
        x = F.relu(self.conv1(x, edge_index))



        #Different feature aggregation
        if self.mode == 'cat':
            xs = [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                xs += [x]

            x = torch.stack(xs, -1).view(bs, num_nodes, -1)

        elif self.mode == 'base':
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

        elif self.mode == 'sum':
            for conv in self.convs:
                x = x + F.relu(conv(x, edge_index))

        # import pdb; pdb.set_trace();

        #Pathway pooling
        x = x.view(bs, -1, x.size(-1))  # [b_size, n_nodes, features]
        x = x.transpose(0, 1)
        x = scatter_mean(x[self.row], self.col, dim=0).view(bs, self.n_cmt, -1) #[b_size, n_cmt, features]



        #Global pooling
        xs = []
        for i in range(x.size(0)):

            if self.graph_agg_mode == 'mean':
                xs.append(global_mean_pool(x[i,:,:], batch[:self.n_cmt])) #batch is an auxiliary batch object (the original one is for n_genes)

            elif self.graph_agg_mode == 'add':
                xs.append(global_add_pool(x[i,:,:], batch[:self.n_cmt])) #batch is an auxiliary batch object (the original one is for n_genes)

            elif self.graph_agg_mode == 'max':
                xs.append(global_max_pool(x[i,:,:], batch[:self.n_cmt])) #batch is an auxiliary batch object (the original one is for n_genes)

            elif self.graph_agg_mode == 'set2set':
                xs.append(self.set2setpooling(x[i,:,:], batch[:self.n_cmt])) #batch is an auxiliary batch object (the original one is for n_genes)


        x = torch.stack(xs, 0).view(bs, -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)
