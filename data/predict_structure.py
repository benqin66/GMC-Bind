import os
import numpy as np
import RNA
from tqdm import tqdm
from scipy import sparse
import re
import glob

# Set ViennaRNA thread count via environment variable
os.environ['VRNA_NR_THREADS'] = '4'


def preprocess_sequence(sequence):
    """Preprocess RNA sequence: convert to uppercase, replace T with U, filter invalid characters"""
    sequence = sequence.upper().replace('T', 'U')
    sequence = re.sub(r'[^AUCG]', 'N', sequence)
    if 'N' in sequence:
        sequence = sequence.replace('N', 'A')
    return sequence


def calculate_base_pairing_probs(sequence):
    """Calculate base pairing probability matrix using ViennaRNA"""
    if len(sequence) < 2:
        return np.zeros((1, 1))

    try:
        md = RNA.md()
        md.max_bp_span = min(150, len(sequence))
        fc = RNA.fold_compound(sequence, md)
        fc.pf()
        bpp_matrix = fc.bpp()
        return np.array(bpp_matrix)

    except Exception as e:
        print(f"Calculation failed: {e}, returning zero matrix")
        return np.zeros((len(sequence), len(sequence)))


def calculate_adjacency_matrix(bpp_matrix):
    """Generate sparse adjacency matrix from base pairing probability matrix"""
    if bpp_matrix.shape[0] == 0 or bpp_matrix.shape[1] == 0:
        return sparse.csr_matrix((1, 1))

    triu = np.triu(bpp_matrix)
    triu[triu < 1e-5] = 0
    return sparse.csr_matrix(triu)


def pad_adjacency_matrix(matrix, max_length):
    """
    Robust adjacency matrix padding function
    """
    # Handle empty matrix
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return sparse.csr_matrix((max_length, max_length))

    # Get original dimensions
    orig_rows, orig_cols = matrix.shape

    # Ensure original matrix is square or close to square
    if orig_rows != orig_cols:
        # If not square, create square matrix with maximum side length
        orig_size = max(orig_rows, orig_cols)
        square_matrix = sparse.csr_matrix((orig_size, orig_size), dtype=matrix.dtype)
        # Ensure new matrix doesn't exceed original dimensions
        rows_to_copy = min(orig_rows, orig_size)
        cols_to_copy = min(orig_cols, orig_size)
        square_matrix[:rows_to_copy, :cols_to_copy] = matrix[:rows_to_copy, :cols_to_copy]
        matrix = square_matrix
        orig_rows = orig_cols = orig_size

    if orig_rows == max_length:
        return matrix

    # Create new padded matrix
    padded_matrix = sparse.csr_matrix((max_length, max_length), dtype=matrix.dtype)

    # Calculate actual size to copy
    copy_size = min(orig_rows, max_length)

    # Copy original matrix data to top-left corner of new matrix
    padded_matrix[:copy_size, :copy_size] = matrix[:copy_size, :copy_size]

    return padded_matrix


def process_fasta_file(input_path, output_dir):
    """Process single FASTA file: read sequences, predict structures, save adjacency matrices"""
    sequences = []
    current_seq = ""
    with open(input_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    sequences.append((header, current_seq))
                    current_seq = ""
                header = line.strip()[1:]  # Remove '>'
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append((header, current_seq))

    if not sequences:
        print(f"No sequences found in file: {input_path}")
        return 0, 0

    # Calculate maximum sequence length in file
    seq_lengths = [len(seq) for _, seq in sequences]
    max_length_in_file = max(seq_lengths)

    all_matrices = []
    sequence_lengths = []

    for header, seq in tqdm(sequences, desc=f"Processing {os.path.basename(input_path)}"):
        # Preprocess sequence
        clean_seq = preprocess_sequence(seq)
        current_length = len(clean_seq)

        # Calculate base pairing probabilities
        bpp_matrix = calculate_base_pairing_probs(clean_seq)

        # Generate adjacency matrix
        adj_matrix = calculate_adjacency_matrix(bpp_matrix)

        # Pad adjacency matrix to maximum length
        padded_matrix = pad_adjacency_matrix(adj_matrix, max_length_in_file)

        # Store results
        all_matrices.append(padded_matrix)
        sequence_lengths.append(current_length)

    # Prepare output filename
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npz")

    # Stack all padded adjacency matrices
    stacked_matrices = sparse.vstack(all_matrices)

    # Save all matrices and sequence length information
    np.savez_compressed(output_path,
                        matrix=stacked_matrices,
                        lengths=np.array(sequence_lengths),
                        max_length=max_length_in_file)

    # Return statistics
    return len(sequences), max_length_in_file


def main():
    # Set paths
    input_dir = "./GraphProt_CLIP_sequences"
    output_dir = "./secondary_structure"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all FASTA files
    fasta_files = glob.glob(os.path.join(input_dir, "**", "*.fa"), recursive=True)

    print(f"Found {len(fasta_files)} FASTA files to process")
    print(f"ViennaRNA version: {RNA.__version__}")

    # Process each file
    summary = []
    for filepath in tqdm(fasta_files, desc="Overall progress"):
        filename = os.path.basename(filepath)
        print(f"\nStarting processing: {filename}")

        try:
            seq_count, max_len = process_fasta_file(filepath, output_dir)
            summary.append({
                'file': filename,
                'seq_count': seq_count,
                'max_len': max_len
            })
            print(f"Processing completed: {filename}, sequences: {seq_count}, max length: {max_len}nt")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing file {filepath}: {str(e)}")

    # Print processing summary
    print("\n==== Processing Summary ====")
    for item in summary:
        print(f"{item['file']}: {item['seq_count']} sequences, max_length={item['max_len']}nt")

    total_seqs = sum(item['seq_count'] for item in summary)
    max_length = max(item['max_len'] for item in summary) if summary else 0
    print(f"\nTotal: {total_seqs} sequences, maximum sequence length={max_length}nt")


if __name__ == "__main__":
    print(f"ViennaRNA version: {RNA.__version__}")
    print("Setting thread count via environment variable: VRNA_NR_THREADS=4")

    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("All processing completed!")
