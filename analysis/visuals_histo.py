import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import csv

def plot_score_distributions(original_scores, generated_scores, column_name="pred_fit"):

   plt.figure(figsize=(8, 5))  # Adjust figure size as needed

   min_score, max_score = 0, 3

    # Filter scores within the zoom range
   filtered_original_scores = [score for score in original_scores if min_score <= score <= max_score]
   filtered_generated_scores = [score for score in generated_scores if min_score <= score <= max_score]
   
   plt.hist(filtered_original_scores, bins=10, edgecolor='black', alpha=0.5, label = "Original Scores")
   plt.xlabel("Scores")
   plt.ylabel("Frequency")
   plt.title("Distribution of Original Sequence Scores")

   plt.hist(filtered_generated_scores, bins=10, edgecolor='black', alpha=0.5, label = "Generated Scores")
   plt.xlabel("Scores")
   plt.ylabel("Frequency")
   plt.title("Distribution of Generated Sequence Scores")
   plt.legend()
   plt.show()


def plot_score_evolution(scores, degree=1):
   """
   Plots the evolution of scores and fits a polynomial curve.
   """

   # Generate x-values
   x = range(1, len(scores) + 1)

   # Fit polynomial curve
   coefficients = np.polyfit(x, scores, degree)
   poly = np.poly1d(coefficients)

   # Generate values for the curve
   x_values = np.linspace(min(x), max(x), 100)
   y_values = poly(x_values)

   # Plot the data and polynomial curve
   plt.plot(x, scores, color='blue', label='Data')
   plt.plot(x_values, y_values, color='red', label='Polynomial Fit (Degree {})'.format(degree))

   # Customize plot
   plt.xlabel('Steps')
   plt.ylabel('Fitness')
   plt.title('Evolution of Scores')
   plt.legend()
   plt.grid(True)
   plt.show()

if __name__ == '__main__':
    
    # Read original scores from TSV file, handling potential errors
    with open('/home/selindemirturk/PAB1_GFP_task_emb/PAB1_GFP_task/data/tournament_evoplay.tsv', 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # Skip header row
        original_scores = []
        for row in reader:
            try:
                score = float(row[1])
                original_scores.append(score)
            except (ValueError, IndexError): continue
                

    # Read generated scores from CSV file
    generated_data = pd.read_csv("/home/selindemirturk/PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/trial/trial.csv")
    generated_scores = generated_data["pred_fit"]

    # Plot score distributions
    plot_score_distributions(original_scores, generated_scores)