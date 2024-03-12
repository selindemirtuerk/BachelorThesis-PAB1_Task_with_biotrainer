import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import csv
import shutil

class FastaConverter:

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def convert_to_fasta(self):
        if self.input_file.endswith(".txt"):
            self._convert_from_txt()
        elif self.input_file.endswith(".csv"):
            self._convert_from_csv()
        elif self.input_file.endswith(".tsv"):
            self._convert_from_tsv()
        elif self.input_file.endswith(".fasta"):
            shutil.copyfile(self.input_file, self.output_file)
        else:
            print("Unsupported file format. Please provide a TXT or CSV file.")

    def _decide_set(self):
        # Example logic to randomly assign sequences to a set
        # Adjust this method to suit your actual criteria for train, val, and test sets
        return random.choice(['train', 'val', 'test'])

    def _convert_from_txt(self):
        with open(self.input_file, "r") as infile, open(self.output_file, "w") as outfile:
            next(infile)
            seq_counter = 1  # Counter to number sequences sequentially
            for line in infile:
                parts = line.strip().split('\t')  # Split line into parts by tab
                if len(parts) < 2:  # Skip any lines that don't have at least two parts
                    continue
                sequence, score = parts[0], parts[1]
                set_assignment = self._decide_set()
                header = f"Seq{seq_counter} TARGET={score} SET={set_assignment}"
                seq_record = SeqRecord(Seq(sequence), id=header, description="")
                SeqIO.write(seq_record, outfile, "fasta")
                seq_counter += 1
        

    def _convert_from_csv(self):
        # Assuming comma as delimiter for CSV files
        with open(self.input_file, mode='r') as infile, open(self.output_file, mode='w') as outfile:
            next(infile)
            csv_reader = csv.reader(infile, delimiter=",")
            seq_counter = 1  # Counter to number sequences sequentially
            for row in csv_reader:
                sequence, score = row
                set_assignment = self._decide_set()
                header = f">Seq{seq_counter} TARGET={score} SET={set_assignment}"
                seq_record = SeqRecord(Seq(sequence), id=header, description="")
                SeqIO.write(seq_record, outfile, "fasta")
                seq_counter += 1
    
    def _convert_from_tsv(self):
        # Assuming comma as delimiter for CSV files
        with open(self.input_file, mode='r') as infile, open(self.output_file, mode='w') as outfile:
            next(infile)
            csv_reader = csv.reader(infile, delimiter="\t")
            seq_counter = 1  # Counter to number sequences sequentially
            for row in csv_reader:
                sequence, score = row
                set_assignment = self._decide_set()
                header = f"Seq{seq_counter} TARGET={score} SET={set_assignment}"
                seq_record = SeqRecord(Seq(sequence), id=header, description="")
                SeqIO.write(seq_record, outfile, "fasta")
                seq_counter += 1