# PAB1_Task_with_biotrainer

## Evoplay with Biotrainer Integration

This is the repository for the Bachelor's Degree Thesis/Project carried out during the 2023/24 Winter Semester by Selin Demirtürk.

It provides a modified version of [EvoPlay](https://github.com/melobio/EvoPlay)'s PAB1_GFP_task, which was first published in [Nature](https://www.nature.com/articles/s42256-023-00691-9). This version is integrated with the open-source tool [BioTrainer](https://github.com/sacdallago/biotrainer), simplifying the process of training machine learning models for biological applications. Its main purpose is to make the use of EvoPlay more accessible to individuals with little to no background in machine learning. Here we focused solely on the PAB1 task.

## Code and Neural Networks

This repository contains all the code developed for the thesis, as well as a link to the the raw results obtained and the trained and used models. Following is the code not implemented by the author and the neural networks used are referenced.

**Code:**

* **Base Code of the Task**. Obtained from the [official repository](https://github.com/melobio/EvoPlay/tree/main/code/PAB1_GFP_task), maintained by model's authors.
* **Biotrainer**. Obtained from the [official repository](https://github.com/sacdallago/biotrainer)

**Neural Networks:**

* **Predictor Neural Network**. Generated with the help of [Biotrainer](https://github.com/sacdallago/biotrainer) using a custom [configuration file](https://github.com/selindemirtuerk/BachelorThesis-PAB1_Task_with_biotrainer/blob/main/oracle_training/config.yml)

* **Policy Value Neural Network**. Provided by [RostLab](https://www.cs.cit.tum.de/en/bio/home/)

## Environment Requirements

**Conda:**

An environment yml file for dependencies can be found [here](https://github.com/selindemirtuerk/BachelorThesis-PAB1_Task_with_biotrainer/blob/main/environment/environment.yml)

```bash
# In the base directory:
conda env create -f /environment/environment.yml
```

**Poetry:**

Alternatively you can use the poetry files in the base directory for the environment set-up

1. Make sure you have [poetry](https://python-poetry.org/) installed: 
```bash
curl -sSL https://install.python-poetry.org/ | python3 - --version 1.4.2
```

2. Install dependencies `poetry`:
```bash
# In the base directory
poetry install
```

## Running

You will need to provide the path to your dataset, number of desired sequences and the name to your desired output directory in order to run the script. Supported file formats are .txt, .csv, .tsv and .fasta

1. With Conda:

```bash
# In the base directory:
python train_m_single_m_p_pab1.py <dataset_path> <num_sequences> <output_dir>
```

Where:
- <dataset_path> is the path to the dataset file you want to use.
- <num_sequences> is the number of sequences to generate (must be an integer).
- <output_dir> is the name of the directory where you want the output to be saved.

2. With Poetry:

```bash
# In the base directory:
poetry python train_m_single_m_p_pab1.py <dataset_path> <num_sequences> <output_dir>
```

## Results

The pipeline produces a csv file containing the desired number of sequences and their respective scores, similar to the original EvoPlay PAB1-GFP-Task. Additionally, we provide a file highlighting the top 10 sequences generated during the execution, a graph depicting the evolutionary trajectory of the scores associated with the generated sequences, another graph contrasting the distribution of the scores of the generated sequences with those of the original dataset.

 

