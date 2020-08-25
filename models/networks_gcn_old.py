import torch
from torch_geometric.nn import TopKPooling, global_mean_pool, global_add_pool, global_max_pool, Set2Set
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch.nn import Linear, BatchNorm1d
from models.conv import GCNConv
from models.conv.cheb_conv import ChebConv
from models.conv.sage_conv import SAGEConv
import torchbnn as bnn
import numpy as np
###########
## this networks use a custom implementation of GCN (by SW) to use only one edge_index
## the DataLoader is also a custom one -- we don't need the edge_index now for the samples or the forward function
## edge_index is passed when constructing the model
## DataLoader provides samples with dimensions [nbatch, ngenes, in_channels]
## This one is directly compatible with the input dimension of data required by captum





########Base models - GCN
#FC
class GCNModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, n_genes = 15135, mode='cat', bayesian = False, batchnorm = False, do_layers = 1):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = GCNConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.do_layers =do_layers

        self.mode = mode

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



    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)


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

        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)


        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)


#Pathways + FC
class GCNModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat', bayesian = False, batchnorm = False, num_genes = 15135, do_layers = 1):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers

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



    def forward(self,  x,  batch):
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
                 bayesian=False, batchnorm=False, k_hops=2, do_layers=1):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = ChebConv(n_features, hidden_gcn, k_hops)
        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.do_layers = do_layers

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

    def forward(self, x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        # First GCN layer
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
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index)).view(bs * num_nodes, -1)).view(bs, num_nodes,
                                                                                                       -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index))

        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)

        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)

#Pathways + FC
class ChebModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat',
                 bayesian=False, batchnorm=False, num_genes=15135, k_hops=2, do_layers=1):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers

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

    def forward(self, x, batch):
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)

        ######  GCNs

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
                    x = x + self.bnconvs[i](F.relu(conv(x, edge_index)).view(bs * num_nodes, -1)).view(bs, num_nodes,
                                                                                                       -1)
                    i += 1
                else:
                    x = x + F.relu(conv(x, edge_index))

        # Pathway pooling
        x = x.view(bs, -1, x.size(-1))  # [b_size, n_nodes, features]
        x = x.transpose(0, 1)
        x = scatter_mean(x[self.row], self.col, dim=0)
        x = x.transpose(0, 1).contiguous().view(-1, self.fc.weight.size(1))

        # FC to aggregate pathway features
        x = self.fc(x).view(-1, self.lin1.weight.size(1))

        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)  # dropout
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)





########GraphSage models

#FC
class SageModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, edge_index, n_genes = 15135, mode='cat', bayesian = False, batchnorm = False, do_layers = 1):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = SAGEConv(n_features, hidden_gcn)
        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.do_layers =do_layers

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



    def forward(self,  x, batch):
        # import pdb; pdb.set_trace();
        # x input dim [nbatch, nnodes, nfeatures]
        edge_index = self.edge_index
        bs, num_nodes = x.size(0), x.size(1)


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




        # FC to aggregate pathway features
        x = self.fc(x).view(bs, num_nodes)
        if self.do_layers == 2:
            x = F.dropout(x, p=0.5, training=self.training)  # dropout

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)



#Pathways + FC
class SageModelWPathways(torch.nn.Module):
    def __init__(self, n_features, n_classes, num_layers, hidden_gcn, hidden_fc, pathway, n_cmt, edge_index, mode='cat', bayesian = False, batchnorm = False, num_genes = 15135, do_layers = 1):
        super().__init__()
        self.edge_index = edge_index
        self.row, self.col = pathway
        self.n_cmt = n_cmt
        self.mode = mode
        self.batchnorm = batchnorm
        self.do_layers = do_layers

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



    def forward(self,  x,  batch):
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
