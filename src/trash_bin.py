# File containing not working approaches


# def get_gap_edge_loss(adj_list, edge_bal_coeff, classification, vertex_cut_coeff, embs_batch, nodes_batch, num_classes,
#                       device, tensorboard, epoch=-1, step=-1, num_steps=-1):
#     '''
#     This loss is not working, because objective function is not differentiable. It's here just for the sake of test.
#     '''
#     node2index = {n: i for i, n in enumerate(nodes_batch)}
#     batch_adj_list = get_reduced_adj_list(nodes_batch, adj_list)
#     edges = [(src, dst) for src in batch_adj_list for dst in batch_adj_list[src]]
#     edges_embeddings = []
#     for src, dst in edges:
#         src_emb = embs_batch[node2index[src]]
#         dst_emb = embs_batch[node2index[dst]]
#         # print("SRC EMB SIZE: ", src_emb.size())
#         edge_emb = torch.cat([src_emb, dst_emb], 0)
#         # print("EDGE EMB SIZE: ", edge_emb.size())
#         edges_embeddings.append(edge_emb)
#     edges_embeddings = torch.stack(edges_embeddings)
#     logits = classification(edges_embeddings)
#     print(logits)
#     nodes_assignments = [[0 for _ in range(num_classes)] for _ in range(len(nodes_batch))]
#     _, edges_assignments = torch.max(logits, 1)
#     print(edges_assignments)
#     for i, e in enumerate(edges):
#         assignment = edges_assignments[i]
#         nodes_assignments[node2index[e[0]]][assignment.item()] = 1
#         nodes_assignments[node2index[e[1]]][assignment.item()] = 1
#
#     nodes_assignments = torch.tensor(nodes_assignments, dtype=torch.float, requires_grad=True)
#     print(nodes_assignments)
#     num_nodes_classes = torch.sum(nodes_assignments, dtype=torch.float)
#     print("NODES CLASSES: ", num_nodes_classes)
#
#     num_nodes = len(nodes_batch)
#     print("NUM NODES: ", num_nodes)
#     num_edges = len(edges)
#     vertex_cut = torch.div(num_nodes_classes, torch.tensor(num_nodes, dtype=torch.float))
#     print("VERTEX CUT: ", vertex_cut)
#     cluster_size = torch.sum(logits, dim=0).to(device)
#     ground_truth = torch.tensor([num_edges / float(num_classes)] * num_classes).to(device)
#     mse_loss = torch.nn.modules.loss.MSELoss()
#     bal = mse_loss(ground_truth, cluster_size)
#
#     # print("Bal: ", bal)
#     # / len(nodes_batch) makes it the same regardless of window size
#     # loss = (cut_coeff * left_sum + bal_coeff * bal) / len(nodes_batch)
#     # print("edge cut: ", edge_cut)
#     loss = (vertex_cut_coeff * vertex_cut + edge_bal_coeff * bal)
#
#     if step != -1:
#         global_step = epoch * num_steps + step + 1
#         tensorboard.add_scalar("vertex_cut", vertex_cut.item(), global_step=global_step)
#         tensorboard.add_scalar("edge_balance", bal.item(), global_step=global_step)
#         tensorboard.add_scalar("edge_gap_loss", loss.item(), global_step=global_step)
#     return loss
#
# def train_gap_edge(nodes_ids, features, graphSage, classification, ds, adj_list, num_classes, device, tensorboard,
#                    b_sz=0,
#                    epochs=800, cut_coeff=1, bal_coeff=1):
#     print('Training GAP Edge...')
#     c_optimizer = torch.optim.Adam(classification.parameters(), lr=7.5e-5)
#     # train classification, detached from the current graph
#     # classification.init_params()
#     train_nodes = nodes_ids
#
#     for epoch in range(epochs):
#         train_nodes = shuffle(train_nodes)
#         if b_sz <= 0:
#             b_sz = len(train_nodes)
#         batches = math.ceil(len(train_nodes) / b_sz)
#         visited_nodes = set()
#         for index in range(batches):
#             nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]
#             visited_nodes |= set(nodes_batch)
#             embs = get_gnn_embeddings(graphSage, nodes_batch, features, adj_list)
#             emb_id_to_node_id = []
#             for emd_id, node_id in enumerate(nodes_batch):
#                 emb_id_to_node_id.append(emd_id)
#
#             embs_batch = embs[emb_id_to_node_id]
#
#             loss = get_gap_edge_loss(adj_list, bal_coeff, classification, cut_coeff, embs_batch, nodes_batch,
#                                      num_classes,
#                                      device, epoch=epoch, step=index + 1, num_steps=batches, tensorboard=tensorboard)
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch + 1, epochs, index,
#                                                                                             batches, loss.item(),
#                                                                                             len(visited_nodes),
#                                                                                             len(train_nodes)))
#             loss.backward()
#
#             nn.utils.clip_grad_norm_(classification.parameters(), 5)
#             c_optimizer.step()
#             c_optimizer.zero_grad()
#
#         # max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
#
#     return classification
