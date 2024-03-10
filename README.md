# PAB1_Task_with_biotrainer

## Evoplay with Biotrainer Integration

This is the repository for the Bachelor's Degree Thesis/Project carried out during the 2023/24 Winter Semester by Selin Demirtürk.

It provides a modified version of [EvoPlay](https://github.com/melobio/EvoPlay)'s PAB1_GFP_task, which was first published in [Nature](https://www.nature.com/articles/s42256-023-00691-9). This version is integrated with the open-source tool [BioTrainer](https://github.com/sacdallago/biotrainer), simplifying the process of training machine learning models for biological applications. Its main purpose is to make the use of EvoPlay more accessible to individuals with little to no background in machine learning.

## Code and Models

This repository contains all the code developed for the thesis, as well as the raw results obtained and the trained and used models. Following is the code not implemented by the author and the pretrained models used are referenced.

**Code:**

* **Base Code of the Task**. Obtained from the [official repository](https://github.com/melobio/EvoPlay/tree/main/code/PAB1_GFP_task), mantained by model's authors.
* **Biotrainer**. Provided by [RostLab](https://www.cs.cit.tum.de/en/bio/home/).

**Models:**


## Usage

You will need to provide the path to your dataset and number of desired sequences in order to run the script. Supported file formats are .txt, .csv, .tsv

```bash
cd PAB1_GFP_Task_with_biotrainer/
python train_m_single_m_p_pab1.py /path/to/your/dataset number_of_sequences_to_generate

```

## Results


