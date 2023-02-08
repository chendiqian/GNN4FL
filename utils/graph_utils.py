import torch
from typing import List
from torch_geometric.data import HeteroData


def my_hetero_collate(graphs: List[HeteroData]) -> HeteroData:
    """
    Self-defined collate function for hetero data.

    :param graphs:
    :return:
    """
    edge_keys = list(graphs[0].edge_index_dict.keys())
    node_keys = sorted(list(graphs[0].x_dict.keys()))

    nedges = torch.tensor([g.edge_index_dict[edge_keys[0]].shape[1] for g in graphs])
    nnodes_aggregator = torch.tensor([g.x_dict[node_keys[0]].shape[0] for g in graphs])
    nnodes_features = torch.tensor([g.x_dict[node_keys[-1]].shape[0] for g in graphs])

    aggregated_node_dict = {k: torch.cat([g[k]['x'] for g in graphs], dim=0) for k in node_keys}

    cumsum_edge = torch.cumsum(
        torch.hstack(
            [torch.tensor([0], dtype=nedges.dtype),
             nedges]
        ),
        dim=0)
    cumsum_nnodes_aggregator = torch.cumsum(
        torch.hstack(
            [torch.tensor([0], dtype=nnodes_aggregator.dtype),
             nnodes_aggregator]
        ),
        dim=0)
    cumsum_nnodes_features = torch.cumsum(
        torch.hstack(
            [torch.tensor([0], dtype=nnodes_features.dtype),
             nnodes_features]),
        dim=0)
    edge_index_rel_a = torch.repeat_interleave(cumsum_nnodes_aggregator[:-1],
                                               nedges,
                                               dim=0)
    edge_index_rel_f = torch.repeat_interleave(cumsum_nnodes_features[:-1],
                                               nedges,
                                               dim=0)

    edge_index_rel = {('clients', 'to', 'aggregator'): torch.vstack([edge_index_rel_f, edge_index_rel_a]),
                      ('aggregator', 'to', 'clients'): torch.vstack([edge_index_rel_a, edge_index_rel_f]),}

    aggreagated_edge_index_dict = {k: torch.cat([g[k]['edge_index'] for g in graphs], dim=1) + edge_index_rel[k] for k in edge_keys}

    extra_attr_dict = {'y': torch.cat([g.y for g in graphs], dim=0),
                       'ptr_feature': cumsum_nnodes_features,
                       'ptr_aggregator': cumsum_nnodes_aggregator,
                       'ptr_edge': cumsum_edge}

    new_graph = HeteroData(**extra_attr_dict)
    for k, v in aggregated_node_dict.items():
        new_graph[k].x = v
    for k, v in aggreagated_edge_index_dict.items():
        new_graph[k].edge_index = v
    return new_graph


def separate(graph: HeteroData, idx: int):
    assert hasattr(graph, 'ptr_feature')
    assert hasattr(graph, 'ptr_aggregator')
    assert hasattr(graph, 'ptr_edge')

    extracted_node_dict = {}
    for k, v in graph.x_dict.items():
        if k == 'aggregator':
            extracted_node_dict[k] = v[graph.ptr_aggregator[idx] : graph.ptr_aggregator[idx + 1]]
        else:
            extracted_node_dict[k] = v[graph.ptr_feature[idx] : graph.ptr_feature[idx + 1]]

    extracted_edge_dict = {}
    for k, v in graph.edge_index_dict.items():
        edge_index = v[:, graph.ptr_edge[idx] : graph.ptr_edge[idx + 1]]
        if k == ('clients', 'to', 'aggregator'):
            rel = torch.tensor([[graph.ptr_feature[idx]],
                                [graph.ptr_aggregator[idx]]])
        elif k == ('aggregator', 'to', 'clients'):
            rel = torch.tensor([[graph.ptr_aggregator[idx]],
                                [graph.ptr_feature[idx]]])
        else:
            raise KeyError
        extracted_edge_dict[k] = edge_index - rel

    new_graph = HeteroData(y=graph.y[graph.ptr_feature[idx] : graph.ptr_feature[idx + 1]])
    for k, v in extracted_node_dict.items():
        new_graph[k].x = v
    for k, v in extracted_edge_dict.items():
        new_graph[k].edge_index = v
    return new_graph
