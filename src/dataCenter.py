import sys
import os

from collections import defaultdict
import numpy as np


class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_dataSet(self, dataSet='cora'):
        adj_list_train = defaultdict(set)
        adj_list_test = defaultdict(set)
        adj_list_val = defaultdict(set)
        adj_list = defaultdict(set)
        if dataSet == 'cora':
            cora_content_file = self.config['file_path.cora_content']
            cora_cite_file = self.config['file_path.cora_cite']

            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            label_map = {}  # map label to Label_ID
            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])
                    node_map[info[0]] = i
                    if not info[-1] in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)

            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
            with open(cora_cite_file) as fp:
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
            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_feats_train', feat_data_train)
            setattr(self, dataSet + '_feats_test', feat_data_test)
            setattr(self, dataSet + '_feats_val', feat_data_val)

            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_list', adj_list)
            setattr(self, dataSet + '_adj_list_train', adj_list_train)
            setattr(self, dataSet + '_adj_list_test', adj_list_test)
            setattr(self, dataSet + '_adj_list_val', adj_list_val)
        elif dataSet == 'fb':
            fb_edges_file = self.config['file_path.fb_edges']
            fb_feats_file = self.config['file_path.fb_feats']

            feat_data = []
            node_map = {}  # map node to Node_ID
            with open(fb_feats_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:]])
                    node_map[info[0]] = i
            feat_data = np.asarray(feat_data)

            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
            with open(fb_edges_file) as fp:
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
            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_feats_train', feat_data_train)
            setattr(self, dataSet + '_feats_test', feat_data_test)
            setattr(self, dataSet + '_feats_val', feat_data_val)

            setattr(self, dataSet + '_adj_list', adj_list)
            setattr(self, dataSet + '_adj_list_train', adj_list_train)
            setattr(self, dataSet + '_adj_list_test', adj_list_test)
            setattr(self, dataSet + '_adj_list_val', adj_list_val)

        # elif dataSet == 'pubmed':
        #     pubmed_content_file = self.config['file_path.pubmed_paper']
        #     pubmed_cite_file = self.config['file_path.pubmed_cites']
        #
        #     feat_data = []
        #     labels = []  # label sequence of node
        #     node_map = {}  # map node to Node_ID
        #     with open(pubmed_content_file) as fp:
        #         fp.readline()
        #         feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        #         for i, line in enumerate(fp):
        #             info = line.split("\t")
        #             node_map[info[0]] = i
        #             labels.append(int(info[1].split("=")[1]) - 1)
        #             tmp_list = np.zeros(len(feat_map) - 2)
        #             for word_info in info[2:-1]:
        #                 word_info = word_info.split("=")
        #                 tmp_list[feat_map[word_info[0]]] = float(word_info[1])
        #             feat_data.append(tmp_list)
        #
        #     feat_data = np.asarray(feat_data)
        #     labels = np.asarray(labels, dtype=np.int64)
        #
        #     adj_lists = defaultdict(set)
        #     with open(pubmed_cite_file) as fp:
        #         fp.readline()
        #         fp.readline()
        #         for line in fp:
        #             info = line.strip().split("\t")
        #             node1 = node_map[info[1].split(":")[1]]
        #             node2 = node_map[info[-1].split(":")[1]]
        #             adj_lists[node1].add(node2)
        #             adj_lists[node2].add(node1)
        #
        #     assert len(feat_data) == len(labels) == len(adj_lists)
        #     test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
        #
        #     setattr(self, dataSet + '_test', test_indexs)
        #     setattr(self, dataSet + '_val', val_indexs)
        #     setattr(self, dataSet + '_train', train_indexs)
        #
        #     setattr(self, dataSet + '_feats', feat_data)
        #     setattr(self, dataSet + '_labels', labels)
        #     setattr(self, dataSet + '_adj_lists', adj_lists)

    def _split_data(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size):]

        return test_indexs, val_indexs, train_indexs
