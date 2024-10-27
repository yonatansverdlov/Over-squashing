import itertools
import math
import random
from typing import List

import numpy as np
import torch
import torch_geometric

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Dataset


class RadiusProblemGraphs(object):
    def __init__(self, args):
        self.args = args
        self.depth = args.depth

    def generate_data(self, train_fraction):
        raise NotImplementedError

    def split_dataset(self, dataset):
        self.args.in_dim, self.args.out_dim = self.get_dims()
        X_train, X_test, X_val = torch.utils.data.random_split(dataset, [5000, 500, 500])
        return X_train, X_test, X_val


class TreeDataset(RadiusProblemGraphs):
    def __init__(self, args):
        super(TreeDataset, self).__init__(args=args)
        self.samples = args.num_samples
        self.depth = args.depth
        self.args = args
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

        X_train, X_test = train_test_split(
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
        in_dim = len(self.leaf_indices)
        out_dim = len(self.leaf_indices) + 1
        return in_dim, out_dim


class RingDataset(TreeDataset):
    def __init__(self, args, add_crosses=False, classes=5):
        super(RingDataset, self).__init__(args=args)
        self.add_crosses = add_crosses
        self.classes = classes

    def generate_ring_transfer_graph(self, nodes, target_label, add_crosses: bool):
        """
        Generate a ring transfer graph with an option to add crosses.

        Args:
        - nodes (int): Number of nodes in the graph.
        - target_label (list): Label of the target node.
        - add_crosses (bool): Whether to add cross edges in the ring.

        Returns:
        - Data: Torch geometric data structure containing graph details.
        """
        assert nodes > 1, ValueError("Minimum of two nodes required")
        # Determine the node directly opposite to the source (node 0) in the ring
        opposite_node = nodes // 2

        # Initialise feature matrix with a uniform feature.
        # This serves as a placeholder for features of all nodes.
        x = np.ones(nodes) 

        # Set feature of the source node to 0 and the opposite node to the target label
        x[0] = self.classes + 1
        x[opposite_node] = target_label

        # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
        x = torch.tensor(x, dtype=torch.int)

        # List to store edge connections in the graph
        edge_index = []
        for i in range(nodes - 1):
            # Regular connections that make the ring
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])

            # Conditionally add cross edges, if desired
            if add_crosses and i < opposite_node:
                # Add edges from a node to its direct opposite
                edge_index.append([i, nodes - 1 - i])
                edge_index.append([nodes - 1 - i, i])

                # Extra logic for ensuring additional "cross" edges in some conditions
                if nodes + 1 - i < nodes:
                    edge_index.append([i, nodes + 1 - i])
                    edge_index.append([nodes + 1 - i, i])

        # Close the ring by connecting the last and the first nodes
        edge_index.append([0, nodes - 1])
        edge_index.append([nodes - 1, 0])

        # Convert edge list to a torch tensor
        edge_index = np.array(edge_index, dtype=int).T
        edge_index = torch.tensor(edge_index, dtype=int)

        # Create a mask to identify the target node in the graph. Only the source node (index 0) is marked true.
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[0] = 1

        # Determine the graph's label based on the target label. This is a singular value indicating the index of the target label.
        y = torch.tensor([np.argmax(target_label)], dtype=int)

        # Return the graph with nodes, edges, mask and the label
        return Data(x=x, edge_index=edge_index, val_mask=mask, y=target_label,train_mask = mask, test_mask = mask)

    def generate_data(self, split):
        """
        Generate a dataset of ring transfer graphs.

        Args:
        - nodes (int): Number of nodes in each graph.
        - add_crosses (bool): Whether to add cross edges in the ring.
        - classes (int): Number of different classes or labels.
        - samples (int): Number of graphs in the dataset.

        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = 2 * self.depth
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = self.args.num_samples // self.classes
        for i in range(self.args.num_samples):
            label = i // samples_per_class
            target_class = np.zeros(self.classes)
            target_class[label] = 1.0
            graph = self.generate_ring_transfer_graph(nodes=nodes, target_label=label, add_crosses=self.add_crosses)
            dataset.append(graph)

        X_train, X_test, X_val = self.split_dataset(dataset=dataset)
        return X_train, X_test, X_val

    def get_dims(self):
        return self.classes + 1, 5


class CliqueRing(TreeDataset):
    def __init__(self, args, classes=5):
        super(CliqueRing, self).__init__(args=args)
        self.classes = classes

    def generate_lollipop_transfer_graph(self, nodes: int, target_label: List[int]):
        """
        Generate a lollipop transfer graph.

        Args:
        - nodes (int): Total number of nodes in the graph.
        - target_label (list): Label of the target node.

        Returns:
        - Data: Torch geometric data structure containing graph details.
        """
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        # Initialize node features. The first node gets 0s, while the last gets the target label
        x = np.ones(nodes) 
        #
        x[0] = 0
        x[nodes - 1] = target_label

        # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
        x = torch.tensor(x, dtype=torch.int)
        #

        edge_index = []

        # Construct a clique for the first half of the nodes,
        # where each node is connected to every other node except itself
        for i in range(nodes // 2):
            for j in range(nodes // 2):
                if i == j:  # Skip self-loops
                    continue
                edge_index.append([i, j])
                edge_index.append([j, i])

        # Construct a path (a sequence of connected nodes) for the second half of the nodes
        for i in range(nodes // 2, nodes - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])

        # Connect the last node of the clique to the first node of the path
        edge_index.append([nodes // 2 - 1, nodes // 2])
        edge_index.append([nodes // 2, nodes // 2 - 1])

        # Convert the edge index list to a torch tensor
        edge_index = np.array(edge_index, dtype=int).T
        edge_index = torch.tensor(edge_index, dtype=int)

        # Create a mask to indicate the target node (in this case, the first node)
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[0] = 1

        # Convert the one-hot encoded target label to its corresponding class index

        return Data(x=x, edge_index=edge_index, root_mask=mask, y=target_label,val_mask = mask,
        train_mask = mask, test_mask = mask)

    def generate_data(self, split):
        """
        Generate a dataset of lollipop transfer graphs.

        Args:
        - nodes (int): Total number of nodes in each graph.
        - classes (int): Number of different classes or labels.
        - samples (int): Number of graphs in the dataset.

        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = 2 * (self.depth - 1)
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = self.samples // self.classes
        for i in range(self.samples):
            label = i // samples_per_class
            target_class = np.zeros(self.classes)
            target_class[label] = 1.0
            graph = self.generate_lollipop_transfer_graph(nodes, label)
            dataset.append(graph)

        X_train, X_test, X_val = self.split_dataset(dataset=dataset)

        return X_train, X_test, X_val

    def get_dims(self):
        return self.classes, 5