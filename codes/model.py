import torch as th
import torch.nn as nn
import torch.nn.functional as F

#from MyGAT import GATConv
#from MySageConv import SAGEConv
from FunctionConv import FunctionConv
#from MyGIN import GINConv

from time import time


class GAT(nn.Module):
    def __init__(
        self,
        include,
        device,
        in_dim,
        hidden_dim,
        out_dim,
        num_heads,
        n_layers,
        dropout,
        combine_type='mean',
        activation=th.nn.functional.relu,
        aggregation_type='mean'
    ):
        super(GAT, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads=num_heads
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GATConv(in_dim, hidden_dim,num_heads[0],combine_type='concat',activation=activation,allow_zero_in_degree=True),
        )
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    num_heads[0],
                    combine_type='concat',
                    activation=activation,
                    allow_zero_in_degree=True
                )
            )
        # output layer
        self.layers.append(GATConv(hidden_dim,out_dim,num_heads=num_heads[1],combine_type='mean',allow_zero_in_degree=True))

    def forward(self, blocks, features):
        h = features

        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            #print(layer.fc)
            h = layer(blocks[i], h)
            #print(h.shape)
            #print(h.shape)
        return h.squeeze(1)

class GIN(nn.Module):
    def __init__(
        self,
        include,
        device,
        in_dim,
        hidden_dim,
        out_dim,
        num_heads,
        n_layers,
        dropout,
        combine_type=None,
        activation=th.nn.functional.relu,
        aggregation_type='mean',
    ):
        super(GIN, self).__init__()
        self.activation = activation
        #print(n_layers)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.fc_init = nn.Linear(in_dim,hidden_dim)
        # input layer
        #print(n_layers)
        in_dim = hidden_dim

        # hidden layers
        for i in range(n_layers):
            self.layers.append(
                GINConv(
                    in_dim,
                    hidden_dim,
                    include = include,
                    aggregator_type=aggregation_type,

                )
            )
            in_dim = hidden_dim
        # output layer

        self.layers.append(GINConv(in_dim, out_dim,include=include,aggregator_type=aggregation_type))

    def forward(self, blocks, features):
        #print("h:",features.shape)
        #h = features
        h = self.activation(self.fc_init(features))

        for i in range(self.n_layers + 1):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](blocks[i], h)

        return h.squeeze(1)


class FuncGCN(nn.Module):
    def __init__(
        self,
        ntypes,         #
        in_dim,
        hidden_dim,
        out_dim,
        dropout,
        activation=th.relu,

    ):
        super(FuncGCN, self).__init__()
        self.activation = activation
        self.ntypes = ntypes
        self.out_dim = out_dim
        self.dropout = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(hidden_dim,out_dim)

        self.conv = FunctionConv(
                    ntypes,
                    in_dim,
                    hidden_dim,
                    activation=activation,
                )


    def forward(self, blocks, features):
        r"""

        Description
        -----------
        forward computation of FGNN

        Parameters
        ----------
        blocks : [dgl_block]
            blocks gives the sampled neighborhood for a batch of target nodes.
            Given a target node n, its sampled neighborhood is organized in layers
            depending on the distance to n.
            A block is a graph that describes the part between two succesive layers,
            consisting of two sets of nodes: the *input* nodes and *output* nodes.
            The output nodes of the last block are the target nodes (POs), and
            the input nodes of the first block are the PIs.
        feature : torch.Tensor
            It represents the input (PI) feature of shape :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of target nodes (POs).

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output (PO) feature.
        """
        depth = len(blocks)
        h = features
        r"""
        The message passes through the blocks layer by layer, from the PIs of the blocks to the POs
        In each iteration, messages are only passed between two successive layers (blocks)
        """
        for i in range(depth):
            if i != 0:
                h = self.dropout(h)

            # we do not need activation function in the last iteration
            act_flag = (i != depth - 1)
            h = self.conv(act_flag, blocks[i], h)
        if self.hidden_dim!=self.out_dim:
            h= self.fc_out(h)
        return h.squeeze(1)


class GCN(nn.Module):
    def __init__(
        self,
        label,
        include,
        device,
        in_dim,
        hidden_dim,
        out_dim,
        num_heads,
        n_layers,
        dropout,
        combine_type='sum',
        activation=th.relu,
        aggregation_type='mean',
    ):
        super(GCN, self).__init__()
        self.activation = activation
        #print(n_layers)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.fc_init = nn.Linear(in_dim,hidden_dim)
        in_dim = hidden_dim
        # input layer
        #print(n_layers)
        # if n_layers>0:
        #     self.layers.append(
        #         SAGEConv(in_dim, hidden_dim, include=include,combine_type = 'sum',aggregator_type=aggregation_type, activation=activation)
        #     )
        # hidden layers
        for i in range(n_layers):
            self.layers.append(
                SAGEConv(
                    in_dim,
                    hidden_dim,
                    include=include,
                    label = label,
                    combine_type = 'sum',
                    aggregator_type=aggregation_type,
                    activation=activation,
                )
            )
            in_dim = hidden_dim
        # output layer

        self.layers.append(SAGEConv(in_dim, out_dim,include=include,label = label,combine_type='sum', aggregator_type=aggregation_type))

    def forward(self, blocks, features):
        #print("h:",features.shape)
        h = features
        #print(blocks)
        #print(features.shape,self.fc_init.weight.shape)
        h = self.activation(self.fc_init(features))
        runtimes = []
        for i in range(self.n_layers + 1):
            if i != 0:
                h = self.dropout(h)
            start = time()
            h = self.layers[i](True,blocks[i], h)
        #     runtime = time()-start
        #     runtimes.append(runtime)
        # print(runtimes)
        return h.squeeze(1)


class BiClassifier(nn.Module):
    def __init__(
        self, GCN1,GCN2, out_dim,n_fcn,combine_type='concat',activation=th.relu,dropout=0.5,
    ):
        super(BiClassifier, self).__init__()
        self.activation = activation
        self.GCN1 = GCN1
        self.GCN2 = GCN2
        self.dropout = nn.Dropout(p=dropout)
        #self.layers1 = self.GCN1.layers
        #self.layers2 = self.GCN2.layers
        self.layers3 = nn.ModuleList()
        self.n_fcn = n_fcn
        #self.alpha = tf.Variable(initial_value=0.0,trainable=True)  # 有问�?
        self.combine_type = combine_type
        #if type(self.GCN1).__name__ == 'GNN_1l':
        if self.GCN1 is None or self.combine_type !='concat':
            hidden_dim = self.GCN2.out_dim
        elif self.GCN2 is None:
            hidden_dim = self.GCN1.out_dim
        else:
            hidden_dim = self.GCN1.out_dim+self.GCN2.out_dim

        for i in range(n_fcn-1):
            self.layers3.append(nn.Linear(hidden_dim, int(hidden_dim/2)))
            hidden_dim = int(hidden_dim / 2)
        self.layers3.append(nn.Linear(hidden_dim, out_dim))
        #print(self.layers)
    def forward(self, in_blocks, in_features,out_blocks,out_features):
        if self.GCN1 is None:
            #start = time()
            h = self.GCN2(out_blocks,out_features)
            #print("GCN2 time:", time() - start)
        elif self.GCN2 is None:
            h = self.GCN1(in_blocks,in_features)
        else:
            h = self.GCN1(in_blocks, in_features)
            #start = time()
            rh = self.GCN2(out_blocks, out_features)
            #print("GCN2 time:",time()-start)
            if self.combine_type == 'concat':
                h = th.cat((h, rh), 1)
            elif self.combine_type == 'sum':
                h = h + rh
            elif self.combine_type == 'max':
                h = th.max(h,rh)
            else:
                print('please select a proper bi-combine function')
                assert False
        #print(h.shape)
        for i in range(self.n_fcn-1):
            #h = self.dropout(self.activation(self.layers3[i](h)))
            h = self.dropout(h)
            h = self.layers3[i](self.activation(h))
        #print(h.shape)
        if self.n_fcn >= 1:
            h = self.layers3[-1](h).squeeze(-1)
        #print("aa")
        return h

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,nlayers,activation =nn.ReLU() ,dropout=0.5):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.layers= nn.Sequential()
        dim1 = in_dim
        for i in range(nlayers-1):
            self.layers.add_module('dropout_{}'.format(i+1),self.dropout)
            self.layers.add_module('activation_{}'.format(i+1), self.activation)
            self.layers.add_module('linear_{}'.format(i+1),nn.Linear(dim1, int(dim1/2)))
            dim1 = int(dim1 / 2)
        self.layers.add_module('linear_{}'.format(nlayers),nn.Linear(dim1, out_dim))
    def forward(self,embedding):
        return self.layers(embedding).squeeze(-1)
