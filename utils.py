import sys
import os
import pandas as pd

def parse_arguments():

    if len(sys.argv) <= 3:
        raise ValueError("Please provide the dataset, the desired number of sequences to be generated and the output directory.")
    
    input_file_path = sys.argv[1]
    # Convert the provided path to an absolute path
    absolute_file_path = os.path.abspath(input_file_path)

    num_of_sequences = sys.argv[2]
    output_dir = sys.argv[3]

    try:
        num_of_sequences = int(num_of_sequences)
    except:
        raise ValueError(f"The provided number of integers is not an integer value: {num_of_sequences}")
    
    # Check if the absolute file path exists and is a file
    if not os.path.isfile(absolute_file_path):
        raise FileNotFoundError(f"The provided file path does not exist or is not a file: {absolute_file_path}")
    
    print(f"Using file: {absolute_file_path} as dataset for the oracle training")

    return absolute_file_path, num_of_sequences, output_dir

def extract_starting_seq(file_path):
    
    # Try to infer the delimiter
    if file_path.endswith('.csv'):
        delimiter = ','
    elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
        delimiter = '\t'
    else:
        raise ValueError("Unsupported file format. Please provide a .txt, .csv, or .tsv file.")
    try:
        # Load the file into a DataFrame
        df = pd.read_csv(file_path, delimiter=delimiter, header=None, names=['Sequence', 'Score'])
    except Exception as e:
        raise ValueError(f"Failed to load the file: {e}")
    # Ensure the 'Score' column is numeric and drop any non-numeric rows
    df.dropna(subset=['Score'], inplace=True)
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df[df['Score'] != 0]
    # Find the row with the highest score
    df.to_csv(file_path, sep=delimiter, index=False, header=False)
    best_row = df.loc[df['Score'].idxmax()]
    return best_row['Sequence']