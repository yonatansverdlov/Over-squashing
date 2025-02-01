import itertools
import math
import random
random.seed(0)
import numpy as np
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from easydict import EasyDict


class RadiusProblemGraphs(object):
    def __init__(self, args:EasyDict, add_crosses,num_classes):
        self.args = args
        self.depth = args.depth
        self.num_train_samples = args.num_train_samples
        self.num_test_samples = args.num_test_samples
        self.add_crosses = add_crosses
        self.num_classes = num_classes
        self.repeat = args.repeat

    def generate_sample(self):
        raise NotImplementedError
    
    def generate_dataset(self,num_samples,index:int=0):
        """
        Generate a dataset of lollipop transfer graphs.
        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = 2 * (self.depth)
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = num_samples // self.num_classes
        for i in range(num_samples):
            label = i // samples_per_class
            target_class = np.zeros(self.num_classes)
            target_class[label] = 1.0
            graph = self.generate_sample(nodes, label,self.add_crosses)
            dataset.append(graph)

        return dataset
    
    def generate_data(self):
        self.args.in_dim, self.args.out_dim = self.get_dims()
        X_train = self.generate_dataset(num_samples=self.num_train_samples)
        return X_train, X_train, X_train
    
    def get_dims(self):
        return self.num_classes, self.num_classes

class TreeDataset(RadiusProblemGraphs):
    def __init__(self, args):
        super(TreeDataset, self).__init__(args=args,add_crosses=None,num_classes=None)
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.repeat = args.repeat

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([cur_node, left_child])
            edges.append([right_child, cur_node])
            edges.append([cur_node, right_child])
            stack.append((right_child, max_node))

            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, )
        return edge_index

    def generate_data(self, train_fraction):
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=int)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            data_list.append(Data(x=nodes, edge_index=edge_index, train_mask=root_mask,val_mask = root_mask,
                        test_mask = root_mask, y=label))

        self.args.in_dim, self.args.out_dim = self.get_dims()

        X_train, _ = train_test_split(
            data_list, train_size=train_fraction, shuffle=True, stratify=[data.y for data in data_list])

        return X_train * self.repeat, X_train, X_train

    # Every sub-class should implement the following methods:

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(num_permutations, math.factorial(num_leaves), max_examples // num_leaves)
            permutations = [np.random.permutation(range(1, num_leaves + 1)) for _ in
                            range(per_depth_num_permutations)]
        else:
            permutations = random.sample(list(itertools.permutations(range(1, num_leaves + 1))),
                                         min(num_permutations, math.factorial(num_leaves)))

        return itertools.chain.from_iterable(

            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations)

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [(selected_key, 0)]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = (leaf_num + 1, values[leaf_num])
            else:
                node = (0, 0)
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        # get input and output dims
        dim = len(self.leaf_indices) + 1
        return dim, dim


class RingDataset(RadiusProblemGraphs):
    def __init__(self, args:EasyDict, add_crosses:bool=False, classes:int=5):
        super(RingDataset, self).__init__(args=args,add_crosses=add_crosses,num_classes=classes)

    def generate_sample(self, nodes: int, target_label: int, add_crosses: bool):
        """
        Generate a ring transfer graph with an option to add crosses.

        Args:
        - nodes (int): Number of nodes in the graph.
        - target_label (int): Label of the target node.
        - add_crosses (bool): Whether to add cross edges in the ring.

        Returns:
        - Data: Torch geometric data structure containing graph details.
        """
        assert nodes > 1, "Minimum of two nodes required"

        # Initialize feature matrix and set target node feature
        x = torch.ones(nodes, dtype=torch.long)
        x[nodes // 2] = target_label  # Source node feature
        # Initialize edges with ring structure
        edge_index = [[i, (i + 1) % nodes] for i in range(nodes)]
        edge_index += [[(i + 1) % nodes, i] for i in range(nodes)]

        # Add cross edges if specified
        if add_crosses:
            for i in range(nodes // 2):
                opposite = nodes - 1 - i
                edge_index.extend([[i, opposite], [opposite, i]])

        # Convert edge list to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # Create mask for target node
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[0] = 1  # Only the target node (index 0) is marked true
        # Return the data object
        return Data(x=x, edge_index=edge_index, val_mask=mask, y=target_label, train_mask=mask, test_mask=mask)

class CliquePath(RadiusProblemGraphs):
    def __init__(self, args:EasyDict, classes:int=5):
        super(CliquePath, self).__init__(args=args,num_classes = classes,add_crosses = False)
        self.depth-=1

    def generate_sample(self, nodes: int, target_label: int, _):
        """
        Generate a lollipop transfer graph.

        Args:
        - nodes (int): Total number of nodes in the graph.
        - target_label (int): Label of the target node.

        Returns:
        - Data: Torch geometric data structure containing graph details.
        """
        if nodes <= 1:
            raise ValueError("Minimum of two nodes required")

        # Initialize node features with 1, setting the last node to the target label
        x = torch.ones(nodes, dtype=torch.int)
        x[nodes - 1] = target_label
        # Construct edges for a clique in the first half
        clique_size = nodes // 2
        edge_index = [[i, j] for i in range(clique_size) for j in range(i + 1, clique_size)]
        edge_index += [[j, i] for i, j in edge_index]  # Add reverse edges for bidirectionality

        # Construct path edges for the second half
        path_edges = [[i, i + 1] for i in range(clique_size, nodes - 1)]
        edge_index += path_edges + [[j, i] for i, j in path_edges]  # Add bidirectional path edges

        # Connect the last node of the clique to the first node of the path
        edge_index += [[clique_size - 1, clique_size], [clique_size, clique_size - 1]]

        # Convert edge list to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # Create mask for the target node (node 0 in this example)
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[0] = 1

        return Data(x=x, edge_index=edge_index, root_mask=mask, y=target_label,
                    val_mask=mask, train_mask=mask, test_mask=mask)