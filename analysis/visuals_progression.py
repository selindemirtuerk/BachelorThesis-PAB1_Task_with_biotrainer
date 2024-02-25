import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_polynomial_fit(data_file, score_column, degree=1):
   """
   Reads data from a CSV file, fits a polynomial curve to scores, 
   and plots the data along with the curve.

   Args:
       data_file (str): Path to the CSV file containing the data.
       score_column (str): Name of the column containing the scores.
       degree (int, optional): Degree of the polynomial curve. Defaults to 1.
   """

   # Read data from CSV
   data = pd.read_csv(data_file)
   scores = data[score_column]

   # Generate x-values
   x = range(1, len(scores) + 1)

   # Fit polynomial curve
   coefficients = np.polyfit(x, scores, degree)
   poly = np.poly1d(coefficients)

   # Generate values for the curve
   x_values = np.linspace(min(x), max(x), 100)
   y_values = poly(x_values)

   # Create plot
   plt.plot(x, scores, color='blue', label='Data')
   plt.plot(x_values, y_values, color='red', label='Polynomial Fit (Degree {})'.format(degree))

   # Customize plot
   plt.xlabel('Steps')
   plt.ylabel('Fitness')
   plt.title('Evolution of Scores')

   plt.xlim(0,8000)
   plt.ylim(0,4)
   y_ticks = np.arange(0, 4.1, 0.5)
   labels = [f"{tick}" for tick in y_ticks]
   plt.yticks(y_ticks, labels)
   
   plt.legend()
   plt.grid(True)
   plt.show()

if __name__ == '__main__':
    
    data_file = '/home/selindemirturk/PAB1_GFP_task_Robert_updated/PAB1_GFP_task_Robert/trial/trial_one_hot_modified_8000.csv'
    score_column = 'pred_fit'
    degree = 1  # Adjust the degree as needed

    plot_polynomial_fit(data_file, score_column, degree)
