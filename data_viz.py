import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DataVisualization:
    def __init__(self, result_csv, output_dir):
        self.data = pd.read_csv(result_csv)
        self.output_dir = output_dir
        os.makedirs("results", exist_ok=True)

    def extract_score_column(self, file_path):
        """
        Extracts the score column from a dataset file where the score column is the second column.

        Args:
            file_path (str): Path to the dataset file (txt, csv, tsv).

        Returns:
            pandas.Series: The score column as a pandas Series.
        """
        # Infer delimiter based on file extension
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.txt':
            delimiter = '\t'  # For space-separated values
        elif file_extension == '.csv':
            delimiter = ','
        elif file_extension == '.tsv':
            delimiter = '\t'
        else:
            raise ValueError("Unsupported file format. Only txt, csv, and tsv files are supported.")

        # Read the dataset file into a DataFrame
        df = pd.read_csv(file_path, delimiter=delimiter, header=None)

        # Assuming score column is always the second column (index 1)
        score_column = pd.to_numeric(df.iloc[:, 1], errors='coerce')

        return score_column

    def plot_polynomial_fit(self, score_column, num_of_generated_seqs, degree=2):
        scores = self.data[score_column]

        x = range(1, len(scores) + 1)

        coefficients = np.polyfit(x, scores, degree)
        poly = np.poly1d(coefficients)

        x_values = np.linspace(min(x), max(x), 100)
        y_values = poly(x_values)

        plt.plot(x, scores, color='blue', label='Data', linewidth=0.5)
        plt.plot(x_values, y_values, color='red', label='Polynomial Fit (Degree {})'.format(degree))

        plt.xlabel('Steps')
        plt.ylabel('Fitness')
        plt.title('Evolution of Scores')

        plt.xlim(0, num_of_generated_seqs)
        plt.ylim(0, 4.1)
        y_ticks = np.arange(0, 4.1, 0.5)
        labels = [f"{tick}" for tick in y_ticks]
        plt.yticks(y_ticks, labels)

        plt.legend()
        plt.grid(True)

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(self.output_dir /'evolution_of_scores.pdf', format="pdf", dpi=600)

    def plot_score_distributions(self, original_scores, generated_scores):
        plt.figure(figsize=(8, 5))

        min_score, max_score = 0, 4

        filtered_original_scores = [score for score in original_scores if min_score <= score <= max_score]
        filtered_generated_scores = [score for score in generated_scores if min_score <= score <= max_score]

        plt.hist(filtered_original_scores, bins=10, edgecolor='black', alpha=0.5, label="Original Scores")
        plt.xlabel("Scores")
        plt.ylabel("Frequency")
        plt.title("Distribution of Original Sequence Scores")

        plt.hist(filtered_generated_scores, bins=10, edgecolor='black', alpha=0.5, label="Generated Scores")
        plt.xlabel("Scores")
        plt.ylabel("Frequency")
        plt.title("Distribution of Generated Sequence Scores")
        plt.legend()

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(self.output_dir / 'distribution_of_scores.pdf', format="pdf", dpi=600)
    
    def get_top_sequences(self):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            print("No numeric columns found in the DataFrame.")
            return
        
        # Assuming the first numeric column is the score column
        score_column = numeric_columns[0]

        top_sequences = self.data.nlargest(10, score_column)
        top_sequences.to_csv(self.output_dir / "top_ten_sequences.csv", index=False)
    
    def create_data_visualisations(self, num_of_generated_seqs, original_dataset_file, generated_dataset_file):
        
        score_column_name = self.data.columns[1]
        self.plot_polynomial_fit(score_column_name, num_of_generated_seqs)
        original_scores = self.extract_score_column(original_dataset_file)
        generated_scores = self.extract_score_column(generated_dataset_file)
        self.plot_score_distributions(original_scores, generated_scores)
        self.get_top_sequences()


    