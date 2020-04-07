import sys
import os

from collections import defaultdict
import numpy as np
import pandas as pd
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph


class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_dataSet(self, dataset='cora'):
        if dataset == 'cora':
            cora_content_file = self.config['file_path.cora_content']
            cora_cite_file = self.config['file_path.cora_cite']

            self.prepare_dataset(dataset, cora_content_file, cora_cite_file)
        elif dataset == 'fb':
            fb_edges_file = self.config['file_path.fb_edges']
            fb_feats_file = self.config['file_path.fb_feats']

            self.prepare_dataset(dataset, fb_feats_file, fb_edges_file)

        elif dataset == "reddit":
            adj_list_train, train_indexs, adj_list_test, test_indexs, adj_list_val, val_indexs, feat_data, train_edges, val_edges, test_edges, edge_labels = self.load_reddit_data()
            feat_data_train = feat_data_test = feat_data_val = feat_data

            setattr(self, dataset + '_test', test_indexs)
            setattr(self, dataset + '_val', val_indexs)
            setattr(self, dataset + '_train', train_indexs)
            setattr(self, dataset + '_feats', feat_data)
            setattr(self, dataset + '_feats_train', feat_data_train)
            setattr(self, dataset + '_feats_test', feat_data_test)
            setattr(self, dataset + '_feats_val', feat_data_val)
            setattr(self, dataset + '_adj_list', adj_list_train)
            setattr(self, dataset + '_adj_list_train', adj_list_train)
            setattr(self, dataset + '_adj_list_test', adj_list_test)
            setattr(self, dataset + '_adj_list_val', adj_list_val)
            setattr(self, dataset + '_train_edges', train_edges)
            setattr(self, dataset + '_val_edges', val_edges)
            setattr(self, dataset + '_test_edges', test_edges)
            setattr(self, dataset + '_edge_labels', edge_labels)

        elif dataset == 'pubmed':
            pubmed_content_file = self.config['file_path.pubmed_paper']
            pubmed_cite_file = self.config['file_path.pubmed_cites']

            adj_list_train = defaultdict(set)
            adj_list_test = defaultdict(set)
            adj_list_val = defaultdict(set)
            adj_list_train = defaultdict(set)
            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            with open(pubmed_content_file) as fp:
                fp.readline()
                feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
                for i, line in enumerate(fp):
                    info = line.split("\t")
                    node_map[info[0]] = i
                    labels.append(int(info[1].split("=")[1]) - 1)
                    tmp_list = np.zeros(len(feat_map) - 2)
                    for word_info in info[2:-1]:
                        word_info = word_info.split("=")
                        tmp_list[feat_map[word_info[0]]] = float(word_info[1])
                    feat_data.append(tmp_list)

            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

            adj_list_train = defaultdict(set)
            with open(pubmed_cite_file) as fp:
                fp.readline()
                fp.readline()
                for line in fp:
                    info = line.strip().split("\t")
                    node1 = node_map[info[1].split(":")[1]]
                    node2 = node_map[info[-1].split(":")[1]]
                    adj_list_train[node1].add(node2)
                    adj_list_train[node2].add(node1)
                    if node1 in train_indexs and node2 in train_indexs:
                        adj_list_train[node1].add(node2)
                        adj_list_train[node2].add(node1)
                    if node1 in test_indexs and node2 in test_indexs:
                        adj_list_test[node1].add(node2)
                        adj_list_test[node2].add(node1)
                    if node1 in val_indexs and node2 in val_indexs:
                        adj_list_val[node1].add(node2)
                        adj_list_val[node2].add(node1)

            assert len(feat_data) == len(labels) == len(adj_list_train)

            setattr(self, dataset + '_test', test_indexs)
            setattr(self, dataset + '_val', val_indexs)
            setattr(self, dataset + '_train', train_indexs)

            setattr(self, dataset + '_feats', feat_data)
            setattr(self, dataset + '_labels', labels)
            setattr(self, dataset + '_adj_lists', adj_list_train)

    def prepare_dataset(self, dataset, feats_file, edges_file):
        adj_list_train = defaultdict(set)
        adj_list_test = defaultdict(set)
        adj_list_val = defaultdict(set)
        adj_list = defaultdict(set)
        feat_data = []
        node_map = {}  # map node to Node_ID
        with open(feats_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data.append([float(x) for x in info[1:]])
                node_map[info[0]] = i
        feat_data = np.asarray(feat_data)
        test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
        with open(edges_file) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                assert len(info) == 2
                node1 = node_map[info[0]]
                node2 = node_map[info[1]]
                adj_list[node1].add(node2)
                adj_list[node2].add(node1)
                if node1 in train_indexs and node2 in train_indexs:
                    adj_list_train[node1].add(node2)
                    adj_list_train[node2].add(node1)
                if node1 in test_indexs and node2 in test_indexs:
                    adj_list_test[node1].add(node2)
                    adj_list_test[node2].add(node1)
                if node1 in val_indexs and node2 in val_indexs:
                    adj_list_val[node1].add(node2)
                    adj_list_val[node2].add(node1)
        for i in train_indexs:
            if i not in adj_list_train.keys():
                adj_list_train[i] = set()
        for i in test_indexs:
            if i not in adj_list_test.keys():
                adj_list_test[i] = set()
        for i in val_indexs:
            if i not in adj_list_val.keys():
                adj_list_val[i] = set()
        feat_data_train = feat_data[train_indexs]
        feat_data_test = feat_data[test_indexs]
        feat_data_val = feat_data[val_indexs]
        setattr(self, dataset + '_test', test_indexs)
        setattr(self, dataset + '_val', val_indexs)
        setattr(self, dataset + '_train', train_indexs)
        setattr(self, dataset + '_feats', feat_data)
        setattr(self, dataset + '_feats_train', feat_data_train)
        setattr(self, dataset + '_feats_test', feat_data_test)
        setattr(self, dataset + '_feats_val', feat_data_val)
        setattr(self, dataset + '_adj_list', adj_list)
        setattr(self, dataset + '_adj_list_train', adj_list_train)
        setattr(self, dataset + '_adj_list_test', adj_list_test)
        setattr(self, dataset + '_adj_list_val', adj_list_val)

    def load_reddit_data(self, normalize=True, load_walks=False, prefix="reddit"):
        reddit_edges = pd.read_csv(self.config['file_path.reddit_edges'])
        reddit_edges_val = pd.read_csv(self.config['file_path.reddit_edges_val'])
        reddit_edges_test = pd.read_csv(self.config['file_path.reddit_edges_test'])
        reddit_feats = np.load(self.config['file_path.reddit_feats'])
        reddit_edge_labels = pd.read_csv(self.config['file_path.reddit_edge_labels'])

        adj_train = defaultdict(set)
        adj_val = defaultdict(set)
        adj_test = defaultdict(set)
        train_nodes = set()
        val_nodes = set()
        test_nodes = set()
        train_edges = []
        val_edges = []
        test_edges = []
        edge_labels = {}
        for i, r in reddit_edges.iterrows():
            e = r["edge"]
            src = int(e.split(",")[0][1:])
            dst = int(e.split(",")[1][1:-1])
            train_edges.append([src, dst])
            adj_train[src].add(dst)
            train_nodes.add(src)
            train_nodes.add(dst)
        for i, r in reddit_edges_val.iterrows():
            e = r["edge"]
            src = int(e.split(",")[0][1:])
            dst = int(e.split(",")[1][1:-1])
            val_edges.append([src, dst])
            adj_val[src].add(dst)
            val_nodes.add(src)
            val_nodes.add(dst)
        for i, r in reddit_edges_test.iterrows():
            e = r["edge"]
            src = int(e.split(",")[0][1:])
            dst = int(e.split(",")[1][1:-1])
            test_edges.append([src, dst])
            adj_test[src].add(dst)
            test_nodes.add(src)
            test_nodes.add(dst)
        for i, r in reddit_edge_labels.iterrows():
            src = r["src"]
            dst = r["dst"]
            label = r["label"]

            edge_labels[(src, dst)] = label

        train_nodes = list(train_nodes)
        val_nodes = list(val_nodes)
        test_nodes = list(test_nodes)
        if normalize and reddit_feats is not None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(reddit_feats)
            reddit_feats = scaler.transform(reddit_feats)

        return adj_train, train_nodes, adj_test, test_nodes, adj_val, val_nodes, reddit_feats, train_edges, val_edges, test_edges, edge_labels  # class_map

    def _split_data(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size):]

        return test_indexs, val_indexs, train_indexs
