import json
from collections import defaultdict

import numpy as np
import pandas as pd


class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def get_train_features(self, dataset):
        if dataset == "reddit":
            reddit_feats = np.load(self.config['file_path.reddit_feats'])
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(reddit_feats)
            reddit_feats = scaler.transform(reddit_feats)
            return reddit_feats

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

        elif dataset == "twitch":
            twitch_feature_file = self.config['file_path.twitch_feats']
            twitch_edge_file = self.config['file_path.twitch_edges']
            twitch_edge_labels_file = self.config['file_path.twitch_edge_labels']

            twitch_val_feature_file = self.config['file_path.twitch_val_feats']
            twitch_val_edge_file = self.config['file_path.twitch_val_edges']

            twitch_test_feature_file = self.config['file_path.twitch_test_feats']
            twitch_test_edge_file = self.config['file_path.twitch_test_edges']

            adj_list_train, train_features, train_edges, train_labels = self.get_twitch_data(twitch_edge_file,
                                                                                             twitch_feature_file,
                                                                                             twitch_edge_labels_file)
            adj_list_val, val_features, val_edges, _ = self.get_twitch_data(twitch_val_edge_file,
                                                                            twitch_val_feature_file)
            adj_list_test, test_features, test_edges, _ = self.get_twitch_data(twitch_test_edge_file,
                                                                               twitch_test_feature_file)

            setattr(self, dataset + '_train', [int(n) for n in adj_list_train.keys()])
            setattr(self, dataset + '_feats_train', train_features)
            setattr(self, dataset + '_adj_list_train', adj_list_train)

            setattr(self, dataset + '_val', [int(n) for n in adj_list_val.keys()])
            setattr(self, dataset + '_feats_val', val_features)
            setattr(self, dataset + '_adj_list_val', adj_list_val)

            setattr(self, dataset + '_test', [int(n) for n in adj_list_test.keys()])
            setattr(self, dataset + '_feats_test', test_features)
            setattr(self, dataset + '_adj_list_test', adj_list_test)

            setattr(self, dataset + '_train_edges', train_edges)
            setattr(self, dataset + '_val_edges', val_edges)
            setattr(self, dataset + '_test_edges', test_edges)

            setattr(self, dataset + '_edge_labels', train_labels)

        elif dataset == 'deezer':
            deezer_feature_file = self.config['file_path.deezer_feats']
            deezer_edge_file = self.config['file_path.deezer_edges']
            deezer_edge_labels_file = self.config['file_path.deezer_edge_labels']

            deezer_val_feature_file = self.config['file_path.deezer_val_feats']
            deezer_val_edge_file = self.config['file_path.deezer_val_edges']

            deezer_test_feature_file = self.config['file_path.deezer_test_feats']
            deezer_test_edge_file = self.config['file_path.deezer_test_edges']

            adj_list_train, train_features, train_edges, train_labels = self.get_deezer_data(deezer_edge_file,
                                                                                             deezer_feature_file,
                                                                                             deezer_edge_labels_file)
            adj_list_val, val_features, val_edges, _ = self.get_deezer_data(deezer_val_edge_file,
                                                                            deezer_val_feature_file)
            adj_list_test, test_features, test_edges, _ = self.get_deezer_data(deezer_test_edge_file,
                                                                               deezer_test_feature_file)

            setattr(self, dataset + '_train', [int(n) for n in adj_list_train.keys()])
            setattr(self, dataset + '_feats_train', train_features)
            setattr(self, dataset + '_adj_list_train', adj_list_train)

            setattr(self, dataset + '_val', [int(n) for n in adj_list_val.keys()])
            setattr(self, dataset + '_feats_val', val_features)
            setattr(self, dataset + '_adj_list_val', adj_list_val)

            setattr(self, dataset + '_test', [int(n) for n in adj_list_test.keys()])
            setattr(self, dataset + '_feats_test', test_features)
            setattr(self, dataset + '_adj_list_test', adj_list_test)

            setattr(self, dataset + '_train_edges', train_edges)
            setattr(self, dataset + '_val_edges', val_edges)
            setattr(self, dataset + '_test_edges', test_edges)

            setattr(self, dataset + '_edge_labels', train_labels)

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

    def get_deezer_data(self, deezer_edge_file, deezer_feature_file, deezer_edge_labels=None):
        deezer_edges = pd.read_csv(deezer_edge_file)
        deezer_feats = np.load(deezer_feature_file)

        adj_list = defaultdict(set)
        deezer_edges = deezer_edges.values.tolist()
        for e in deezer_edges:
            adj_list[int(e[0])].add(int(e[1]))
            adj_list[int(e[1])].add(int(e[0]))

        edge_labels = {}
        if deezer_edge_labels is not None:
            deezer_edge_labels = pd.read_csv(deezer_edge_labels)
            for i, r in deezer_edge_labels.iterrows():
                src = r["src"]
                dst = r["dst"]
                label = r["label"]

                edge_labels[(src, dst)] = label
                edge_labels[(dst, src)] = label

        return adj_list, deezer_feats, deezer_edges, edge_labels

    def get_twitch_data(self, twitch_edge_file, twitch_feature_file, twitch_edge_labels=None):
        feature_dim = 3170  # maximum feature id of all twitch datasets
        features_raw = json.load(open(twitch_feature_file))
        node2idx = {}
        features = []
        raw_values = [i for v in features_raw.values() for i in v]
        for i, it in enumerate(features_raw.items()):
            node2idx[int(it[0])] = i
            feats = np.zeros(feature_dim)
            feats[[int(val) for val in it[1]]] = 1
            features.append(feats)
        features = np.asarray([np.asarray(f) for f in features])
        edges_raw = pd.read_csv(twitch_edge_file)
        edges = edges_raw.values.tolist()
        edges = list(map(lambda x: [int(x[0]), int(x[1])], edges))
        adj_list = defaultdict(set)
        for e in edges:
            adj_list[node2idx[int(e[0])]].add(node2idx[int(e[1])])
            adj_list[node2idx[int(e[1])]].add(node2idx[int(e[0])])
        edge_labels = {}
        if twitch_edge_labels is not None:
            twitch_edge_labels = pd.read_csv(twitch_edge_labels)
            for i, r in twitch_edge_labels.iterrows():
                src = r["src"]
                dst = r["dst"]
                label = r["label"]

                edge_labels[(src, dst)] = label
                edge_labels[(dst, src)] = label

        return adj_list, features, edges, edge_labels

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
                feat_data.append([float(x) for x in info[1:-1]])
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
