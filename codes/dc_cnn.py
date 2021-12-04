from __future__ import annotations
from typing import List, Dict, Set, Optional
import sys
import os

sys.path.append("..")
#from util.cut_enumeration import Cut, cut_enum
#from util.dc_verilog_parser_baseline import DcParser

import networkx as nx
import random
import copy
import itertools
import collections
import cProfile
import pickle
import time
import torch
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # print("init dataset, len=", len(data))
        # print(torch.max(data))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx][:16], self.data[idx][16])


class EVCNN(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 8), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(64 * (in_dim - 8) / 2), 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        # x = self.bn(x)
        # print(x)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        # print(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)
        # print(x)
        # input()
        return x


def baseline():
    from networkx.algorithms.flow import edmonds_karp

    adders = ["brent_kung", "bounded_fanout", "cond_sum", "hybrid", "ling", "sklansky"]
    for adder in adders:
        fname = adder + "_ev.pkl"
        with open("../GCN/testdata/" + adder + "_adder.pkl", "rb") as f:
            dgl_g = pickle.load(f)
            dgl_gT = dgl_g.reverse(share_ndata=True, share_edata=True)
        if os.path.exists(fname):
            data = torch.load(fname)
        else:
            nxg = g.to_networkx()
            data = []
            max_t = 0
            for i in range(g.num_nodes()):
                ev = torch.clone(g.ndata["ntype"][i]).detach()
                l = nx.single_source_shortest_path_length(nxg, i, cutoff=2)
                for n, d in l.items():
                    ev += g.ndata["ntype"][n] * (1 if d == 1 else 0.5)
                l = nx.single_source_shortest_path_length(
                    nxg.reverse(copy=False), i, cutoff=2
                )
                for n, d in l.items():
                    ev += g.ndata["ntype"][n] * (1 if d == 1 else 0.5)
                max_t = max(max_t, torch.max(ev).item())
                ev = torch.clamp(ev, 0, 64)
                dt = torch.cat((ev, g.ndata["label_i"][i], g.ndata["label_o"][i]))
                data.append(dt)
            data = torch.stack(data)
            print(data.size())
            print(max_t)
            torch.save(data, fname)
            print("saved ", fname)
        max_in_recall = 0
        min_in_recall = 1
        max_out_recall = 0
        min_out_recall = 1
        max_runtime = 0
        min_runtime = 999
        for i in range(95, 100):
            print("###### {}.{} ######".format(adder, i))
            icnn = torch.load(str(i) + "icnn.pkl").cuda()
            ocnn = torch.load(str(i) + "ocnn.pkl").cuda()
            dataset = Dataset(data)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=128, shuffle=False
            )
            pred_i = []
            pred_o = []
            for feature, _, _ in dataloader:
                feature = feature.cuda()
                with torch.no_grad():
                    pred_i.append(torch.argmax(icnn(feature), dim=1))
                    pred_o.append(torch.argmax(ocnn(feature), dim=1))
            pred_in = torch.cat(pred_i, dim=0).cpu().tolist()
            print(len(pred_in))
            pred_out = torch.cat(pred_o, dim=0).cpu().tolist()
            # dgl_g = g  # rename
            g = dgl_g.to_networkx(
                node_attrs=["ntype", "label_i", "label_o"], edge_attrs=[]
            )
            g = nx.DiGraph(g)  # it was multidigraph
            gT = dgl_gT.to_networkx(
                node_attrs=["ntype", "label_i", "label_o"], edge_attrs=[]
            )
            gT = nx.DiGraph(gT)  # it was multidigraph
            in_nodes = set()
            out_nodes = set()
            pred_in_nodes = set()
            pred_out_nodes = set()
            for i, n in enumerate(g.nodes):
                label_i = g.nodes[n]["label_i"]
                pred_i = pred_in[i]
                label_o = g.nodes[n]["label_o"]
                pred_o = pred_out[i]
                if label_i:
                    in_nodes.add(n)
                if label_o:
                    out_nodes.add(n)
                if pred_i:
                    pred_in_nodes.add(n)
                if pred_o:
                    pred_out_nodes.add(n)
            iR = len(in_nodes.intersection(pred_in_nodes)) / len(in_nodes)
            iP = len(in_nodes.intersection(pred_in_nodes)) / len(pred_in_nodes)
            iF = 2 * iR * iP / (iR + iP)
            oR = len(out_nodes.intersection(pred_out_nodes)) / len(out_nodes)
            oP = len(out_nodes.intersection(pred_out_nodes)) / len(pred_out_nodes)
            oF = 2 * oR * oP / (oR + oP)
            print(
                "[ GNN] #in: {}/{}, R={:.3f}, P={:.3f}, #out:{}/{}, R={:.3f}, P={:.3f}".format(
                    len(in_nodes.intersection(pred_in_nodes)),
                    len(in_nodes),
                    iR,
                    iP,
                    len(out_nodes.intersection(pred_out_nodes)),
                    len(out_nodes),
                    oR,
                    oP,
                )
            )
            # hack: add sibling
            hack = time.time()
            new_pred_in_nodes = set()  # copy.deepcopy(pred_in_nodes)
            new_pred_out_nodes = set()  # copy.deepcopy(pred_out_nodes)
            for n in list(pred_in_nodes):
                new_pred_in_nodes.add(n)
                for p in g.successors(n):
                    for k in g.predecessors(p):
                        new_pred_in_nodes.add(k)
            for n in list(pred_out_nodes):
                new_pred_out_nodes.add(n)
                for p in g.predecessors(n):
                    for k in g.successors(p):
                        new_pred_out_nodes.add(k)
            print("hack:{}".format(time.time() - hack))
            cap = time.time()
            g.add_node("S", type="SOURCE")
            g.add_node("T", type="SINK")
            gT.add_node("ST", type="SOURCET")
            gT.add_node("TT", type="SINKT")
            print("reverse graph:{}".format(time.time() - cap))
            for e in g.edges:
                g.edges[e]["capacity"] = 2
            for i in new_pred_in_nodes:
                g.add_edge("S", i, capacity=1)
            for o in pred_out_nodes:
                g.add_edge(o, "T", capacity=2)
            # for e in gT.edges:
            #     gT.edges[e]["capacity"] = 2
            for o in new_pred_out_nodes:
                gT.add_edge("ST", o, capacity=2)
            for i in pred_in_nodes:
                gT.add_edge(i, "TT", capacity=1)
            # gT = g.reverse(copy=False)
            print("add cap:{}".format(time.time() - cap))
            flow1 = time.time()
            flow_value, flow_dict = nx.maximum_flow(g, "S", "T", flow_func=edmonds_karp)
            print("flow1:{}".format(time.time() - flow1))
            flow2 = time.time()
            flow_value, gT_flow_dict = nx.maximum_flow(
                gT, "ST", "TT", flow_func=edmonds_karp
            )
            print("flow2:{}".format(time.time() - flow2))
            # print(flow_value)
            end = time.time()
            print("runtime:{}".format(end - hack))
            flow_in_nodes = set()
            flow_out_nodes = set()
            gT_flow_in_nodes = set()
            gT_flow_out_nodes = set()
            for i, v in flow_dict["S"].items():
                if v > 0 and i in pred_in_nodes:
                    flow_in_nodes.add(i)
            for o in pred_out_nodes:
                if flow_dict[o]["T"] > 0 and o in pred_out_nodes:
                    flow_out_nodes.add(o)
            for o, v in gT_flow_dict["ST"].items():
                if v > 0:
                    gT_flow_out_nodes.add(o)
            for i in pred_in_nodes:
                if gT_flow_dict[i]["TT"] > 0:
                    gT_flow_in_nodes.add(i)
            iR = len(in_nodes.intersection(gT_flow_in_nodes)) / len(in_nodes)
            iP = len(in_nodes.intersection(gT_flow_in_nodes)) / len(gT_flow_in_nodes)
            iF = 2 * iR * iP / (iR + iP)
            oR = len(out_nodes.intersection(flow_out_nodes)) / len(out_nodes)
            oP = len(out_nodes.intersection(flow_out_nodes)) / len(flow_out_nodes)
            oF = 2 * oR * oP / (oR + oP)
            print(
                "[FLOW] #in: {}/{}, R={:.3f}, P={:.3f}, #out:{}/{}, R={:.3f}, P={:.3f}".format(
                    len(in_nodes.intersection(gT_flow_in_nodes)),
                    len(in_nodes),
                    iR,
                    iP,
                    len(out_nodes.intersection(flow_out_nodes)),
                    len(out_nodes),
                    oR,
                    oP,
                )
            )
            max_in_recall = max(max_in_recall, iR)
            min_in_recall = min(min_in_recall, iR)
            max_out_recall = max(max_out_recall, oR)
            min_out_recall = min(min_out_recall, oR)
            max_runtime = max(max_runtime, end - hack)
            min_runtime = min(min_runtime, end - hack)
        print(
            " & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}& {:.1f}$\pm${:.1f}".format(
                (max_in_recall + min_in_recall) / 2,
                (max_in_recall - min_in_recall) / 2,
                (max_out_recall + min_out_recall) / 2,
                (max_out_recall - min_out_recall) / 2,
                (max_runtime + min_runtime) / 2,
                (max_runtime - min_runtime) / 2,
            )
        )


def test(ocnn):
    if os.path.exists("../data/ev/rocket_data.pkl"):
        with open("../data/ev/rocket_data.pkl", "rb") as f:
            data = torch.load(f)
    else:
        with open("../data/simplify9/rocket2.pkl", "rb") as f:
            g = pickle.load(f)
        g.ndata['position'][g.ndata['label_o'].squeeze(-1) == -1] = 100
        unlabel_low(g, 1)
        g.ndata['label_o'][g.ndata['label_o'].squeeze(-1) == 2] = 1
        nids = torch.tensor(range(g.number_of_nodes()))
        nids = nids[g.ndata['label_o'].squeeze(-1) != -1].numpy().tolist()
        print(torch.max(g.ndata["ntype"]))
        nxg = g.to_networkx()
        data = []
        max_t = 0
        for i in nids:
            ev = torch.clone(g.ndata["ntype"][i]).detach()
            l = nx.single_source_shortest_path_length(nxg, i, cutoff=2)
            for n, d in l.items():
                ev += g.ndata["ntype"][n] * (1 if d == 1 else 0.5)
            l = nx.single_source_shortest_path_length(
                nxg.reverse(copy=False), i, cutoff=2
            )
            for n, d in l.items():
                ev += g.ndata["ntype"][n] * (1 if d == 1 else 0.5)
            max_t = max(max_t, torch.max(ev).item())
            ev = torch.clamp(ev, 0, 64)
            dt = torch.cat((ev, g.ndata["label_o"][i]))
            data.append(dt)
        data = torch.stack(data)
        print(data.size())
        print(max_t)
        torch.save(data, "rocket_data.pkl")

    dataset = Dataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
    itp = 0
    itn = 0
    ifp = 0
    ifn = 0
    otp = 0
    otn = 0
    ofp = 0
    ofn = 0
    for feature,  label_o in dataloader:
        feature = feature.cuda()
        label_o = label_o.long().cuda()
        with torch.no_grad():

            ocnn.eval()

            pred_o = torch.nn.functional.softmax(ocnn(feature), dim=1)


            otp += torch.sum(((pred_o[:, 1] >= 0.5) & (label_o == 1))).item()
            otn += torch.sum(((pred_o[:, 0] > 0.5) & (label_o == 0))).item()
            ofp += torch.sum(((pred_o[:, 1] >= 0.5) & (label_o == 0))).item()
            ofn += torch.sum(((pred_o[:, 0] > 0.5) & (label_o == 1))).item()


    orecall = otp / (otp + ofn)
    oprecision = 0 if otp == 0 else otp / (otp + ofp)
    of1 = (
        0
        if orecall + oprecision == 0
        else 2 * orecall * oprecision / (orecall + oprecision)
    )

    print(
        "[TEST, o] recall:{:.3f}, precision:{:.3f}, F1:{:.3f}".format(
            orecall, oprecision, of1
        )
    )
def unlabel_low(g,unlabel_threshold):
    mask_low = g.ndata['position'] <= unlabel_threshold
    g.ndata['label_o'][mask_low] = 0

def train():

    if os.path.exists("../data/ev/boom_data.pkl"):
        data = torch.load("../data/ev/boom_data.pkl")
        print(torch.max(data))
    else:
        with open("../data/simplify9/boom2.pkl", "rb") as f:
            g = pickle.load(f)
        g.ndata['position'][g.ndata['label_o'].squeeze(-1) == -1] = 100
        unlabel_low(g, 1)
        g.ndata['label_o'][g.ndata['label_o'].squeeze(-1) == 2] = 1
        nids = torch.tensor(range(g.number_of_nodes()))
        nids = nids[g.ndata['label_o'].squeeze(-1) != -1].numpy().tolist()
        nxg = g.to_networkx()
        data = []
        max_t = 0
        for i in nids:
            ev = torch.clone(g.ndata["ntype"][i]).detach()
            l = nx.single_source_shortest_path_length(nxg, i, cutoff=2)
            for n, d in l.items():
                ev += g.ndata["ntype"][n] * (1 if d == 1 else 0.5)
            l = nx.single_source_shortest_path_length(
                nxg.reverse(copy=False), i, cutoff=2
            )
            for n, d in l.items():
                ev += g.ndata["ntype"][n] * (1 if d == 1 else 0.5)
            max_t = max(max_t, torch.max(ev).item())
            ev = torch.clamp(ev, 0, 64)
            dt = torch.cat((ev, g.ndata["label_o"][i]))
            data.append(dt)
        data = torch.stack(data)
        print(data.size())
        print(max_t)
        os.makedirs("../data/ev/")
        torch.save(data, "../data/ev/boom_data.pkl")

    dataset = Dataset(data)
    weights = [20 if data[i][16] > 0 else 1 for i in range(data.size(0))]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=sampler,)
    print("batch per epoch:{}".format(len(dataloader)))

    #icnn = EVCNN(16).cuda()
    ocnn = EVCNN(16).cuda()
    #optim1 = torch.optim.Adam(icnn.parameters(), lr=0.0005)
    optim2 = torch.optim.Adam(ocnn.parameters(), lr=0.0005)
    for epoch in range(100):
        #icnn.train()
        ocnn.train()
        i_running_loss = 0
        o_running_loss = 0
        for i, (feature, label_o) in enumerate(dataloader):
            feature = feature.cuda()
            #label_i = label_i.long().cuda()
            label_o = label_o.long().cuda()
            #pred_i = icnn(feature)
            pred_o = ocnn(feature)
            #iloss = nn.functional.cross_entropy(pred_i, label_i)
            oloss = nn.functional.cross_entropy(pred_o, label_o)
            #optim1.zero_grad()
            optim2.zero_grad()
            #iloss.backward()
            oloss.backward()
            #optim1.step()
            optim2.step()
            #i_running_loss += iloss.item()
            o_running_loss += oloss.item()
            if i % 1000 == 999:
                print(
                    "[{}, {}] iloss: {:.3f}, oloss: {:.3f}".format(
                        epoch, i + 1, i_running_loss / 1000, o_running_loss / 1000
                    )
                )
                i_running_loss = 0
                o_running_loss = 0
        if epoch % 5 == 0:
            test( ocnn)
        if epoch >= 95:
            #torch.save(icnn, str(epoch) + "icnn.pkl")
            torch.save(ocnn, str(epoch) + "ocnn.pkl")


if __name__ == "__main__":
    train()
    #baseline()
