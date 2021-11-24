import sys

sys.path.append("..")
import util.structural as structural
import util.verilog as verilog

import dgl
from dgl.data import DGLDataset
import networkx as nx
import torch as th
import numpy as np
import os
import random
from util.new_verilog_parser import DcParser
from options import get_options
from torch.nn.parameter import Parameter


def parse_single_file(parser,vfile_pair,hier_report,label2id):
    # nodes: list of (node, {"type": type}) here node is a str ,like 'n123' or '1'b1'
    # note that here node type does not include buf /not
    # label2id={
    #     "1'b0":0,
    #     "1'b1":1,
    #     'AND':2,
    #     'DFF':3,
    #     'DFFSSR':4,
    #     'DFFAS':5,
    #     'AO':6,
    #     'INV':7,
    #     'OA':8,
    #     'AOI':9,
    #     'NAND':10,
    #     'OAI':11,
    #     'OR':12,
    #     'NOR':13,
    #     'IBUFF':14,
    #     'DELLN':15,
    #     'MUX':16,
    #     'XOR':17,
    #     'XNOR':18,
    #     'NBUFF':19,
    #     'MAJ':20,
    #     'PI':21,
    #     'DFFNASRQ':22,
    #     'DFFN':23,
    #     'DFFAR':24,
    # }
    # label2id ={"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3, 'AO': 4, 'DFFAS': 5, 'OA': 6, 'NAND': 7, 'NBUFF': 8, 'AND': 9,
    #  'IBUFF': 10, 'OR': 11, 'DELLN': 12, 'AOI': 13, 'INV': 14, 'NOR': 15, 'OAI': 16, 'XOR': 17, 'MUX': 18, 'XNOR': 19,
    #  'MAJ': 20, 'PI': 21}
    label2id ={"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3,  'DFFAS': 4, 'NBUFF': 5, 'AND': 6,
     'OR': 7, 'DELLN': 8,  'INV': 9, 'XOR': 10, 'MUX': 11, 'MAJ': 12, 'PI': 13}
    label2id ={"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3,  'DFFAS': 4,  'NAND': 5, 'NBUFF': 6, 'AND': 7,
     'IBUFF': 8, 'OR': 9, 'DELLN': 10,  'INV': 11, 'NOR': 12,  'XOR': 13, 'MUX': 14, 'XNOR': 15,
     'MAJ': 16, 'PI': 17}
    # simplify9
    label2id = {"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3, 'DFFAS': 4, 'NAND': 5, 'AND': 6,
                 'OR': 7, 'DELLN': 8, 'INV': 9, 'NOR': 10, 'XOR': 11, 'MUX': 12, 'XNOR': 13,
                'MAJ': 14, 'PI': 15}
    # simplify10
    # label2id = {"1'b0": 0, "1'b1": 1,  'NAND': 2, 'AND': 3,
    #             'OR': 4, 'DELLN': 5, 'INV': 6, 'NOR': 7, 'XOR': 8, 'MUX': 9, 'XNOR': 10,
    #             'MAJ': 11, 'PI': 12}
    # label2id = {"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFN': 3,  'NAND': 4, 'NBUFF': 5, 'AND': 6,
    #             'IBUFF': 7, 'OR': 8, 'DELLN': 9, 'INV': 10, 'NOR': 11, 'XOR': 12, 'MUX': 13, 'XNOR': 14,
    #             'MAJ': 15, 'PI': 16}

    # label2id ={"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3, 'AO': 4, 'DFFAS': 5, 'OA': 6, 'NAND': 7, 'NBUFF': 8, 'AND': 9,
    #  'IBUFF': 10, 'OR': 11, 'DELLN': 12, 'AOI': 13, 'INV': 14, 'NOR': 15, 'OAI': 16, 'XOR': 17, 'MUX': 18, 'XNOR': 19,
    #  'MAJ': 20, 'PI': 21}
    # label2id = {
    #     "1'b0": 0,
    #     "1'b1": 1,
    #     'AND': 2,
    #     'DFF': 3,
    #     'DFFSSR': 4,
    #     'DFFAS': 5,
    #     'AO': 6,
    #     'INV': 7,
    #     'OA': 8,
    #     'AOI': 9,
    #     'NAND': 10,
    #     'OAI': 11,
    #     'OR': 12,
    #     'NOR': 13,
    #     'IBUFF': 14,
    #     'DELLN': 15,
    #     'MUX': 16,
    #     'XOR': 17,
    #     'XNOR': 18,
    #     'NBUFF': 19,
    #     'MAJ': 20,
    #     'PI': 21,
    # }
    nodes, edges = parser.parse(vfile_pair,hier_report)
    print(len(nodes))
    # nodes,edges = parser.remove_div(nodes,edges)
    # print(len(nodes))
    #print("num_nodes:",len(nodes))
    # print(nodes)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    # adders = structural.find_rca(G)
    # nodes, edges, in_nodes, out_nodes = structural.add_cla_csa(nodes, edges, adders)
    # for n in nodes:
    #     n[1]["is_input"] = 1 if n[0] in in_nodes else 0
    #     n[1]["is_output"] = 1 if n[0] in out_nodes else 0

    # assign an integer id
    node2id = {}
    for n in nodes:
        if node2id.get(n[0]) is None:
            nid = len(node2id)
            node2id[n[0]] = nid
    #print("num_nodes:{}".format(len(node2id)))
    is_adder = th.zeros((len(node2id), 1), dtype=th.long)
    is_adder_input = th.zeros((len(node2id), 1), dtype=th.long)
    is_adder_output = th.zeros((len(node2id), 1), dtype=th.long)
    is_mul_input = th.zeros((len(node2id), 1), dtype=th.long)
    is_mul_output = th.zeros((len(node2id), 1), dtype=th.long)
    is_sub_input = th.zeros((len(node2id), 1), dtype=th.long)
    is_sub_output = th.zeros((len(node2id), 1), dtype=th.long)
    position = th.zeros((len(node2id), 1), dtype=th.long)
    print(label2id)
    for n in nodes:
        nid = node2id[n[0]]
        # print(nid)
        # if label2id.get(n[1]['type']) is None:
        #     print('new type',n[1]['type'])
        #     # if 'DFF' in n[1]['type']:
            #
            # type_id = len(label2id)
            # label2id[n[1]['type']] = type_id
        if get_options().region:
            is_adder[nid][0] = n[1]['is_adder']
        else:
            is_adder_input[nid][0] = n[1]["is_adder_input"]
            is_adder_output[nid][0] = n[1]["is_adder_output"]
            is_mul_input[nid][0] = n[1]["is_mul_input"]
            is_mul_output[nid][0] = n[1]["is_mul_output"]
            is_sub_input[nid][0] = n[1]["is_sub_input"]
            is_sub_output[nid][0] = n[1]["is_sub_output"]
            if n[1]["position"] is not None:
                # if n[1]['is_input']:
                #     print('input',n[1]["position"])
                # elif n[1]['is_output']:
                #     print('output',n[1]["position"])
                # else:
                #     print('false',n[1]["position"])
                position[nid][0] = n[1]["position"][1]
    ntype = th.zeros((len(node2id), get_options().in_dim), dtype=th.float)
    for n in nodes:
        nid = node2id[n[0]]
        if label2id.get(n[1]['type']) is None:
            print('new type', n[1]['type'])
            if  'DFF' in n[1]['type']:
                ntype[nid][2] = 1
        else:
            ntype[nid][label2id[n[1]["type"]]] = 1

    print('muldiv_outputs:',len(is_mul_output[is_mul_output==1]))
    print('muldiv_inputs1:', len(is_mul_input[is_mul_input == 1]))
    print('muldiv_inputs2:', len(is_mul_input[is_mul_input == 2]))
    print('sub_outputs:', len(is_sub_output[is_sub_output == 1]))
    print('sub_inputs1:', len(is_sub_input[is_sub_input == 1]))
    print('sub_inputs2:', len(is_sub_input[is_sub_input == 2]))
    print('adder_outputs:', len(is_adder_output[is_adder_output == 1]))
    print('adder_inputs1:', len(is_adder_input[is_adder_input == 1]))

    src_nodes = []
    dst_nodes = []
    is_reverted = []
    for src, dst, edict in edges:
        src_nodes.append(node2id[src])
        dst_nodes.append(node2id[dst])
        is_reverted.append([0, 1] if edict["is_reverted"] else [1, 0])

    graph = dgl.graph(
        (th.tensor(src_nodes), th.tensor(dst_nodes)), num_nodes=len(node2id)
    )
    for n in nodes:
        nid = node2id[n[0]]
    graph.ndata["ntype"] = ntype

    print('ntype:',ntype.shape)
    #exit()
    if get_options().region:
        graph.ndata["label_ad"] = is_adder
    else:
        graph.ndata['adder_i'] = is_adder_input
        graph.ndata['adder_o'] = is_adder_output
        graph.ndata['mul_i'] = is_mul_input
        graph.ndata['mul_o'] = is_mul_output
        graph.ndata['sub_i'] = is_sub_input
        graph.ndata['sub_o'] = is_sub_output

    graph.edata["r"] = th.FloatTensor(is_reverted)
    graph.ndata['position'] = position

    print("adder input position:",position.squeeze(1)[is_adder_input.squeeze(1) == True])
    print("adder output position:",position.squeeze(1)[is_adder_output.squeeze(1) == True])
    return graph


class Dataset(DGLDataset):
    def __init__(self, top_module,data_paths,report_folders,label2id):
        self.label2id =label2id
        self.data_paths = data_paths
        self.report_folders = report_folders
        self.parser = DcParser(top_module,adder_keywords=['add_x','alu_DP_OP','div_DP_OP'],sub_keywords=['sub_x'],hadd_type="xor")
        super(Dataset, self).__init__(name="dac")
        # self.alpha = Parameter(th.tensor([1]))

    def process(self):
        type2label = {"ling_adder":1,"hybrid_adder":2, "cond_sum_adder":3, "sklansky_adder":4, "brent_kung_adder":5, "bounded_fanout_adder":6,"unknown":7}
        self.batch_graphs = []
        self.graphs = []
        self.len = 0
        vfile_pairs = {}
        for i,path in enumerate(self.data_paths):
            files = os.listdir(path)
            #random.shuffle(files)
            exclude_files = []
            for v in files:
                if not v.endswith('v') or v.split('.')[0].endswith('d10') or 'auto' in v:
                    continue
                if v.startswith('hier'):
                    vname = v[5:-2]
                    vfile_pairs[vname] = vfile_pairs.get(vname, [])
                    vfile_pairs[vname].insert(0, v)
                else:
                    vname = v[:-2]
                    vfile_pairs[vname] = vfile_pairs.get(vname, [])
                    vfile_pairs[vname].append(v)
            vfile_pairs = vfile_pairs.values()
            print(vfile_pairs)
            for vfile_pair in vfile_pairs:
                hier_vf, vf = vfile_pair[0], vfile_pair[1]
                hier_report = os.path.join(self.report_folders[i], hier_vf[:-1] + 'rpt')
                hier_vf = os.path.join(path, hier_vf)
                vf = os.path.join(path, vf)
                # print(case_name)
                # if case_name in ['ut1','ut2','ut3','ut36']:
                #     exclude_files.append(file)
                #     continue
                print("Processing file {}".format(vfile_pair[1]))
                self.len += 1
                graph = parse_single_file(self.parser, (hier_vf,vf), hier_report,self.label2id)
                self.graphs.append(graph)


        self.batch_graph = dgl.batch(self.graphs)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    random.seed(726)
    datapaths = ["../dc/rocket/implementation/"]
    th.multiprocessing.set_sharing_strategy('file_system')
    # print(dataset_not_edge.Dataset_n)
    #dataset = Dataset_dc("Rocket", datapaths, None)
    with open("C:\\Users\\Zeay\\Documents/GitHub\\graph-detection\\dc\\sub\\adder8\\test.v",'r') as f:
        print(f.readlines())
    parser = DcParser('rocket',["alu_DP_OP", "add_x"],'xor')
    parse_single_file(parser,"C:\\Users\\Zeay\\Documents/GitHub\\graph-detection\\dc\\sub\\adder8\\test.v",1,None)

