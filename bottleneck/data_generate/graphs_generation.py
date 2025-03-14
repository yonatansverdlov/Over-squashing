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

def merge_graphs(G1: Data, G2: Data, idx_1: int, idx_2: int, target_label: int):
        """
        Merge two graphs G1 and G2 with the same number of nodes.
        - G1's first node has the label.
        - All other nodes in G1 and G2 are set to 1.
        - G1's node at idx_1 is connected to G2's node at idx_2.
        - Mask is placed on G2's last node.

        Args:
        - G1 (Data): First graph.
        - G2 (Data): Second graph.
        - idx_1 (int): Node index in G1 to be connected.
        - idx_2 (int): Node index in G2 to be connected.
        - target_label (int): Label assigned to the first node of G1.

        Returns:
        - Data: Merged graph with adjusted features, edges, and mask.
        """
        num_nodes = G1.x.size(0)
        
        # Create new feature tensor
        x = torch.ones(2 * num_nodes, dtype=torch.long)
        x[0] = target_label  # First node of G1 gets the label
        
        # Merge edge indices and connect idx_1 in G1 to idx_2 in G2
        edge_index = torch.cat(
            (G1.edge_index, 
             G2.edge_index + num_nodes, 
             torch.tensor([[idx_1, idx_2 + num_nodes], [idx_2 + num_nodes, idx_1]], dtype=torch.long)),
            dim=1
        )
        
        # Define mask on last node of G2
        mask = torch.zeros(2 * num_nodes, dtype=torch.bool)
        mask[num_nodes + G2.most_distant] = 1
        
        return Data(x=x, edge_index=edge_index, root_mask=mask, y=target_label,
                    val_mask=mask, train_mask=mask, test_mask=mask)

class RadiusProblemGraphs(object):
    def __init__(self, args:EasyDict, add_crosses:bool,classes:int):
        self.args = args
        self.depth = args.depth
        self.num_train_samples = args.num_train_samples
        self.num_test_samples = args.num_test_samples
        self.add_crosses = add_crosses
        self.classes = classes
        self.repeat = args.repeat
        self.args = args
        args.classes = classes

    def generate_sample(self):
        raise NotImplementedError
    
    def generate_dataset(self,num_samples):
        """
        Generate a dataset of lollipop transfer graphs.
        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = 2 * (self.depth)
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = num_samples // self.classes
        for i in range(num_samples):
            label = i // samples_per_class
            graph = self.generate_sample(nodes, label,self.add_crosses)
            dataset.append(graph)

        return dataset
    
    def generate_data(self):
        self.args.in_dim, self.args.out_dim = self.get_dims()
        X_train = self.generate_dataset(num_samples=self.num_train_samples)
        return X_train, X_train, X_train
    
    def get_dims(self):
        return self.classes, self.classes
    
    def one_hot_encode(self, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        One-hot encode indices using torch.nn.functional.one_hot
        without `import torch.nn.functional as F`.
        """
        # Use the fully-qualified path directly:
        oh = torch.nn.functional.one_hot(indices, num_classes=num_classes)
        return oh.float()

class TreeDataset(RadiusProblemGraphs):
    def __init__(self, args):
        super(TreeDataset, self).__init__(args=args,add_crosses=None,classes=None)
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
        in_dim, out_dim = self.get_dims()
        self.args.in_dim = in_dim
        self.args.out_dim = out_dim
        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=int)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            nodes = self.one_hot_encode(nodes, in_dim)
            data_list.append(Data(x=nodes, edge_index=edge_index, train_mask=root_mask,val_mask = root_mask,
                        test_mask = root_mask, y=label))

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
        super(RingDataset, self).__init__(args=args,add_crosses=add_crosses,classes=classes)

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
        x[0] = target_label  # Source node feature
        x = self.one_hot_encode(x,self.classes)
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
        source_mask = torch.zeros(nodes, dtype=torch.bool)
        mask[nodes // 2] = 1  # Only the target node (index 0) is marked true
        source_mask[0] = 1
        # Return the data object
        return Data(x=x, edge_index=edge_index, val_mask=mask, y=target_label, train_mask=mask, test_mask=mask,most_distant = nodes//2,
                    source_mask = source_mask,num_gradps = 1)

class CliquePath(RadiusProblemGraphs):
    def __init__(self, args:EasyDict, classes:int=5):
        super(CliquePath, self).__init__(args=args,classes = classes,add_crosses = False)
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
        x = torch.ones(nodes, dtype=torch.long)
        x[nodes - 1] = target_label
        x = self.one_hot_encode(x,self.classes)
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

class TwoConnectedCycles(RingDataset):
    def __init__(self, args: EasyDict, classes: int = 5):
        super(TwoConnectedCycles, self).__init__(args=args, classes=classes, add_crosses=False)

    def generate_sample(self, nodes: int, target_label: int, _):
        """
        Generate a graph consisting of two connected cycles.

        Args:
        - nodes (int): Total number of nodes in the graph (must be even).
        - target_label (int): Label for the source node in C1.

        Returns:
        - Data: Torch geometric data structure containing graph details.
        """
        data_1 = super(TwoConnectedCycles, self).generate_sample(nodes, target_label, False)
        data_2 = super(TwoConnectedCycles, self).generate_sample(nodes, target_label, False)
        return merge_graphs(G1 = data_1, G2 = data_2, target_label = target_label,
                            idx_1 = data_1.most_distant, idx_2 = 0)
    
class PathGraph(RadiusProblemGraphs):
    def __init__(self, args: EasyDict, classes: int = 5):
        super(PathGraph, self).__init__(args=args, classes=classes, add_crosses=False)

    def generate_dataset(self,num_samples):
        """
        Generate a dataset of lollipop transfer graphs.
        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = (self.depth)
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = num_samples // self.classes
        for i in range(num_samples):
            label = i // samples_per_class
            graph = self.generate_sample(nodes, label,self.add_crosses)
            dataset.append(graph)

        return dataset

    def generate_sample(self, nodes: int, target_label: int, _):
        """
        Generate a simple path graph.

        Args:
        - nodes (int): Number of nodes in the graph.
        - target_label (int): Label for the source node.

        Returns:
        - Data: Torch geometric data structure containing graph details.
        """
        if nodes <= 1:
            raise ValueError("Minimum of two nodes required for a path.")

        # Initialize node features with ones and set first node to target label
        x = torch.ones(nodes+1, dtype=torch.long)
        x[0] = target_label

        # Create edges for a simple path
        edge_index = [[i, i + 1] for i in range(nodes)]
        edge_index += [[i + 1, i] for i in range(nodes)]
        
        # Convert edge list to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        
        # Create masks: source (first node) and target (last node)
        mask = torch.zeros(nodes+1, dtype=torch.bool)
        mask[-1] = 1  # Target node at the end of the path
        
        return Data(x=x, edge_index=edge_index, root_mask=mask, y=target_label,
                    val_mask=mask, train_mask=mask, test_mask=mask)

class KIndependentPaths(RadiusProblemGraphs):
    def __init__(self, args: EasyDict, k: int=5, classes: int = 5):
        super(KIndependentPaths, self).__init__(args=args, classes=classes, add_crosses=False)
        self.k = k

    def generate_dataset(self,num_samples):
        """
        Generate a dataset of lollipop transfer graphs.
        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = (self.depth)
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = num_samples // self.classes
        for i in range(num_samples):
            label = i // samples_per_class
            graph = self.generate_sample(nodes, label,self.add_crosses)
            dataset.append(graph)

        return dataset

    def generate_sample(self, nodes: int, target_label: int, _):
        """
        Generate a graph with K independent paths connected to a common start and end node.

        Args:
        - nodes (int): Number of nodes per path.
        - target_label (int): Label for the source node.

        Returns:
        - Data: Torch geometric data structure containing graph details.
        """
        if nodes < 2:
            raise ValueError("Minimum of two nodes required per path.")

        total_nodes = self.k * nodes + 2  # Additional nodes for start and end
        start_node = 0
        end_node = total_nodes - 1
        x = torch.ones(total_nodes, dtype=torch.long)
        x[start_node] = target_label
        
        edge_index = []
        
        for i in range(self.k):
            offset = i * nodes + 1  # Paths start after start_node
            edge_index.append([start_node, offset])  # Connect start to each path
            edge_index.append([offset, start_node])
            
            for j in range(nodes - 1):
                edge_index.append([offset + j, offset + j + 1])
                edge_index.append([offset + j + 1, offset + j])
            
            edge_index.append([offset + nodes - 1, end_node])  # Connect end of path to end node
            edge_index.append([end_node, offset + nodes - 1])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        
        mask = torch.zeros(total_nodes, dtype=torch.bool)
        mask[end_node] = 1
        
        return Data(x=x, edge_index=edge_index, root_mask=mask, y=target_label,
                    val_mask=mask, train_mask=mask, test_mask=mask)
 
class OneRadiusProblemStarGraph(RadiusProblemGraphs):
    def __init__(self, args: EasyDict, add_crosses = False, classes = 10):
        super().__init__(args, add_crosses, classes)

    def generate_sample(self, n:int, label:int, add_crosses:bool):
        """
        Generate a single graph sample with one-hot encoding support.
        """
        num_nodes = n + 1
        center_node = n

        # Create edges
        edge_index = []
        for i in range(n):
            edge_index.append([i, center_node])
            edge_index.append([center_node, i])
                    
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Generate node features for A
        random_ids_A = random.sample(range(n), n)
        labels_A = [random.randint(0, self.classes-1) for _ in range(n)]

        # Here we call one_hot_encode without a separate F import:
        id_one_hot_A = self.one_hot_encode(torch.tensor(random_ids_A), self.n)
        label_one_hot_A = self.one_hot_encode(torch.tensor(labels_A), self.classes + 1)

        center_id = random.randrange(n)
        center_id_one_hot = self.one_hot_encode(torch.tensor([center_id]), self.n)
        # Suppose you want a special label for center node:
        center_label_one_hot = self.one_hot_encode(
            torch.tensor([self.classes]),
            self.classes+1
        )

        x = torch.cat([
            torch.cat([id_one_hot_A, label_one_hot_A], dim=-1),  # A nodes
            torch.cat([center_id_one_hot, center_label_one_hot], dim=-1)
        ], dim=0)

        center_label = labels_A[random_ids_A.index(center_id)]

        # Create mask and target
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[center_node] = True
        y = torch.tensor([center_label], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y,
                train_mask=mask, val_mask=mask, test_mask=mask)
        
    def generate_dataset(self, num_samples):
        """
        Generate a dataset of radius problem graphs.
        - num_samples: Number of graphs to generate.
        Returns:
        - list[Data]: List of PyTorch Geometric Data objects.
        """
        nodes = self.n  # Use self.depth as the size of sets A and B
        if nodes <= 1:
            raise ValueError("Minimum of two nodes required")
        
        dataset = []
        samples_per_class = num_samples // self.classes
        
        for i in range(num_samples):
            label = i // samples_per_class
            graph = self.generate_sample(nodes, label, self.add_crosses)
            dataset.append(graph)

        return dataset
    
    def get_dims(self):
        """
        Get input and output dimensions for the graphs.
        Returns:
        - (in_dim, out_dim): Tuple representing input and output dimensions.
        """
        return self.n + self.classes + 1, self.classes


class TwoRadiusProblemStarGraph(RadiusProblemGraphs):
    def __init__(self, args: EasyDict, add_crosses = False, classes = 10,K=1):
        super().__init__(args, add_crosses, classes)
        self.K = K

    def generate_dataset(self,num_samples):
        """
        Generate a dataset of lollipop transfer graphs.
        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = 2 * (self.depth)
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = num_samples // self.classes
        for i in range(num_samples):
            label = i // samples_per_class
            graph = self.generate_sample()
            dataset.append(graph)

        return dataset

    def generate_sample(self):
        """
        Generate a single graph sample.
        - n: Number of nodes in sets A and B.
        - label: Label for the current sample.
        - add_crosses: (Unused in this implementation, but included for compatibility).
        """
        # Node count: n (A) + 1 (v) + n (B) = 2n + 1
        n = self.args.n
        num_nodes = 2 * n + self.K
        center_nodes = [i for i in range(n, n+self.K)]  # Index of the central node v

        # Create edges:
        edge_index = []
        for i in range(n):  # Edges from A to v and v to B
            for center_node in center_nodes:
                edge_index.append([i, center_node])
                edge_index.append([center_node, i])
                edge_index.append([center_node, i + n + self.K])  
                edge_index.append([i + n + self.K, center_node])  

        # Convert edge_index to tensor (2, num_edges)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 1️⃣ Generate node IDs
        random_ids_A = torch.tensor(random.sample(range(n), n), dtype=torch.long)  # Unique IDs for A
        random_ids_B = random_ids_A[torch.randperm(n)]  # Shuffle to assign B IDs randomly from A
        labels_A = torch.randint(0, self.classes, (n,), dtype=torch.long)  # Random labels for A
        labels_B = torch.full((n,), self.classes)
        id_to_label_map = {id.item(): label.item() for id, label in zip(random_ids_A, labels_A)}

        # 2️⃣ Efficiently retrieve labels for B using dictionary lookup
        y = torch.tensor([id_to_label_map[id.item()] for id in random_ids_B], dtype=torch.long)
        # One-hot encode IDs
        random_ids_A = self.one_hot_encode(random_ids_A, n + self.K)  # (n, n+1)
        random_ids_B = self.one_hot_encode(random_ids_B, n + self.K)  # (n, n+1)
        # One-hot encode labels
        labels_A = self.one_hot_encode(labels_A, self.classes + 1)  # One-hot labels for A
        labels_B = self.one_hot_encode(labels_B, self.classes + 1)  # All B labels = classes+1
        # 3️⃣ Assign labels to B
        feature_v_id = self.one_hot_encode(torch.tensor([n]), n + self.K)
        feature_v_label = self.one_hot_encode(torch.tensor([self.classes]), self.classes + 1)

        # 5️⃣ Combine all features into a single tensor
        x_id = torch.cat((random_ids_A,feature_v_id,random_ids_B,),dim=0)
        x_label = torch.cat((labels_A,feature_v_label,labels_B),dim=0)
        # 7️⃣ Create masks (only B nodes used for training)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[n + self.K:] = True  # Only B nodes included in the mask

        # 8️⃣ Create PyTorch Geometric Data object
        data = Data(x_id=x_id,x_label = x_label, edge_index=edge_index, y=y, train_mask=mask, val_mask=mask, test_mask=mask,
                    x = x_id)

        return data
    def get_dims(self):
        """
        Get input and output dimensions for the graphs.
        Returns:
        - (in_dim, out_dim): Tuple representing input and output dimensions.
        """
        return self.n + 1, self.classes