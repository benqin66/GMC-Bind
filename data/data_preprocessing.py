
import dgl
import torch
import numpy as np
import os
import json
import re
import scipy.sparse as sparse
from dgl.data import DGLDataset
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split


class LoadData(DGLDataset):
    """
    RNA Graph Dataset Loader with Secondary Structure Features
    
    Loads RNA sequence data with corresponding secondary structure information
    and converts them into graph representations for machine learning tasks.
    """

    def __init__(self, dataset_name, config):
        self.config = config
        self.data_path = config.get('data_path', './data/GraphProt_CLIP_sequences/')
        # Fix 1: Ensure correct structure path
        self.struct_path = config.get('struct_path', './data/secondary_structure/')
        self.graphs = []
        self.labels = []
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []
        self.struct_cache = {}  # Cache for loaded structure data
        self.max_seq_length = 0  # Maximum sequence length attribute
        super().__init__(name=dataset_name)

    def process(self):
        """Main processing method for loading and converting RNA data to graphs"""
        # Use data paths directly without subdirectory
        dataset_path = self.data_path
        struct_dataset_path = self.struct_path

        print(f"Processing sub-dataset: {self._name}")
        print(f"Sequence data path: {dataset_path}")
        print(f"Structure data path: {struct_dataset_path}")

        # Define file types and corresponding labels
        file_types = [
            ("train.positives", 1),
            ("train.negatives", 0),
            ("ls.positives", 1),
            ("ls.negatives", 0)
        ]

        all_samples = []

        # Load data from all file types
        for file_suffix, label in file_types:
            filename = f"{self._name}.{file_suffix}.fa"
            file_samples = self._load_fasta(dataset_path, filename, label)

            # Add file type information to each sample
            for seq, _, header in file_samples:
                all_samples.append((seq, label, header, f"{self._name}.{file_suffix}"))

        if not all_samples:
            print(f"Warning: No sequences found in dataset {self._name}")
            return

        labels = [label for _, label, _, _ in all_samples]  # Extract all labels

        # First split: Training set (80%) and temporary set (20%)
        indices = list(range(len(all_samples)))
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Second split: Validation set (10%) and test set (10%)
        val_idx, test_idx, val_labels, test_labels = train_test_split(
            temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        # Calculate maximum sequence length using only training set
        max_length = 0
        for idx in train_idx:
            seq = all_samples[idx][0]
            if len(seq) > max_length:
                max_length = len(seq)
        self.max_seq_length = max_length
        print(f"Dataset {self._name} maximum sequence length (training set only): {max_length}")

        # Create graph structures
        for i, (seq, label, header, file_suffix) in tqdm(enumerate(all_samples), total=len(all_samples),
                                                         desc=f"Building RNA graphs: {self._name}"):
            # Get secondary structure features - pass file type information
            struct_matrix = self._get_structure_matrix(struct_dataset_path, file_suffix, header, len(seq))

            # Create graph
            g = self._sequence_to_graph(seq, struct_matrix, max_length)
            self.graphs.append(g)
            self.labels.append(label)  # Store integer labels directly, not tensors

        print(f"Dataset {self._name} loading completed:")
        print(f"  Training samples: {len(self.train_idx)}")
        print(f"  Validation samples: {len(self.val_idx)}")
        print(f"  Test samples: {len(self.test_idx)}")

    def _load_fasta(self, data_path, filename, label):
        """Load FASTA file and return sequences with labels"""
        filepath = os.path.join(data_path, filename)
        sequences = []

        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist. Using empty data instead.")
            return []

        with open(filepath, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                seq = str(record.seq).upper()
                sequences.append((seq, label, record.description))

        return sequences

    def _get_structure_matrix(self, struct_dataset_path, file_suffix, header, seq_length):
        """Retrieve secondary structure adjacency matrix"""
        # Fix 2: Build correct structure file path
        struct_file = f"{file_suffix}.npz"
        struct_path = os.path.join(self.struct_path, struct_file)  # Use class-defined struct_path

        # Print detailed debug information
        print(f"Dataset: {self._name}")
        print(f"File suffix: {file_suffix}")
        print(f"Structure file path: {struct_path}")
        print(f"File exists: {os.path.exists(struct_path)}")

        # Check if file exists
        if os.path.exists(struct_path):
            print(f"Successfully found structure file: {struct_path}")
        else:
            # Try alternative paths
            alt_path1 = os.path.join("./data/secondary_structure/", struct_file)
            alt_path2 = os.path.join(struct_dataset_path, struct_file)

            print(f"Alternative path 1: {alt_path1} - exists: {os.path.exists(alt_path1)}")
            print(f"Alternative path 2: {alt_path2} - exists: {os.path.exists(alt_path2)}")

            if os.exists(alt_path1):
                struct_path = alt_path1
                print(f"Using alternative path 1: {alt_path1}")
            elif os.path.exists(alt_path2):
                struct_path = alt_path2
                print(f"Using alternative path 2: {alt_path2}")
            else:
                print(f"Error: Structure file {struct_file} not found in any path:")
                print(f"1. {struct_path}")
                print(f"2. {alt_path1}")
                print(f"3. {alt_path2}")
                return sparse.coo_matrix((seq_length, seq_length))

        # Safety check: Sequence length should not exceed training set maximum
        if seq_length > self.max_seq_length and self.max_seq_length > 0:
            print(f"Warning: Sequence length {seq_length} exceeds training set maximum {self.max_seq_length}")

        # Get structure data from cache or load it
        if struct_path in self.struct_cache:
            struct_data = self.struct_cache[struct_path]
        else:
            try:
                print(f"Loading structure file: {struct_path}")
                data = np.load(struct_path, allow_pickle=True)
                struct_data = {
                    'matrix': data['matrix'][()],
                    'lengths': data['lengths'],
                    'max_length': data['max_length']
                }
                self.struct_cache[struct_path] = struct_data
            except Exception as e:
                print(f"Failed to load structure data: {e}")
                return sparse.coo_matrix((seq_length, seq_length))

        # Find matrix index for current sequence
        idx = self._find_sequence_index(struct_data, header, seq_length)
        if idx is not None:
            # Extract corresponding adjacency matrix
            max_length = struct_data['max_length']
            start_idx = idx * max_length
            end_idx = (idx + 1) * max_length

            # Get entire matrix block
            adj_matrix = struct_data['matrix'][start_idx:end_idx, :]

            # Crop to actual sequence length
            actual_length = struct_data['lengths'][idx]
            actual_length = min(actual_length, seq_length)  # Use minimum of actual and current length
            adj_matrix = adj_matrix[:actual_length, :actual_length]

            # Ensure matrix size matches maximum sequence length
            if adj_matrix.shape[0] < self.max_seq_length:
                # Convert matrix to COO format first
                adj_matrix_coo = adj_matrix.tocoo()

                # Get rows, columns and data
                rows, cols = adj_matrix_coo.row, adj_matrix_coo.col
                data = adj_matrix_coo.data

                # Create new matrix
                new_matrix = sparse.coo_matrix(
                    (data, (rows, cols)),
                    shape=(self.max_seq_length, self.max_seq_length)
                )
                return new_matrix

            # Convert to COO format
            return adj_matrix.tocoo()

        print(f"Warning: Sequence {header} index not found in structure file {struct_path}")
        return sparse.coo_matrix((seq_length, seq_length))

    def _find_sequence_index(self, struct_data, header, seq_length):
        """Find sequence index in structure file"""
        # In actual applications, implement logic to match sequences by header
        # Here we use sequence length as a simple matching method
        for idx, len_in_file in enumerate(struct_data['lengths']):
            # If sequence lengths match, assume it's the same sequence
            if len_in_file == seq_length:
                return idx
        return None

    def _sequence_to_graph(self, sequence, struct_matrix, max_length):
        """Convert RNA sequence and secondary structure to graph structure"""

        # Safety check: Ensure input length is reasonable
        if len(sequence) > max_length:
            print(f"Warning: Sequence length {len(sequence)} > maximum length {max_length}")

        # Calculate padding length
        pad_length = max_length - len(sequence)

        # Convert sequence to index representation
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
        indices = [base_to_index.get(base, 4) for base in sequence]  # Unknown bases represented as 4

        # Create node features (one-hot encoding)
        num_nodes = max_length  # Use maximum length
        node_features = torch.zeros(num_nodes, 5)  # 5 features: A, C, G, U, unknown
        for i, idx in enumerate(indices):
            node_features[i, idx] = 1

        # Create graph - linear chain structure
        src_nodes = []
        dst_nodes = []
        edge_features = []

        # Add edges between adjacent bases
        for i in range(len(sequence) - 1):
            src_nodes.append(i)
            dst_nodes.append(i + 1)
            edge_features.append(1.0)  # Adjacent edge feature

            src_nodes.append(i + 1)
            dst_nodes.append(i)  # Bidirectional edge
            edge_features.append(1.0)

        # Add secondary structure edges (using predicted base pairing probabilities)
        if struct_matrix is not None:
            rows, cols, values = struct_matrix.row, struct_matrix.col, struct_matrix.data
            for i, j, prob in zip(rows, cols, values):
                if i < len(sequence) and j < len(sequence):  # Ensure indices are within range
                    # Add bidirectional edges
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    edge_features.append(prob)

                    src_nodes.append(j)
                    dst_nodes.append(i)
                    edge_features.append(prob)

        # Create DGL graph
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)

        # Record original edge count
        original_edge_count = g.number_of_edges()

        # Add self-loops
        g = dgl.add_self_loop(g)

        # Create features for self-loop edges (value 1.0)
        self_loop_features = [1.0] * (g.number_of_edges() - original_edge_count)
        # Combine original edge features and self-loop features
        all_edge_features = edge_features + self_loop_features

        g.ndata['feat'] = node_features
        g.edata['feat'] = torch.tensor(all_edge_features).float().unsqueeze(1)  # Edge weight features

        return g

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def train(self):
        return [self[i] for i in self.train_idx]

    @property
    def val(self):
        return [self[i] for i in self.val_idx]

    @property
    def test(self):
        return [self[i] for i in self.test_idx]

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.tensor(labels, dtype=torch.long)  # Specify long integer type
        return batched_graph, labels

    def _add_self_loops(self):
        """Add self-loops to each node in the graph"""
        for i in range(len(self.graphs)):
            self.graphs[i] = dgl.add_self_loop(self.graphs[i])
