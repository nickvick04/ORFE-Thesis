# ----------------------------------------------------------------------------------------
# This code is designed to clean the lemmatized NAWL .csv file for research purposes
# Original file source: https://www.newgeneralservicelist.com/new-general-service-list-1
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

# necessary package imports
import csv

# given file name from website
INPUT_PATH = "NAWL_1.2_lemmatized_for_research.csv"

def clean_nawl_csv(input_path, output_path):
    '''Function that takes in a file containing multiple columns and outputs the values as 
     a single column to a clean csv file.'''

    word_list = []
    # open, read, and store each word in a list
    with open(input_path, "r", encoding="latin-1") as infile:
        reader = csv.reader(infile)
        for row in reader:
            for word in row:
                w = word.strip() # strip whitespace
                if w:
                    word_list.append(w)

    # write each word as a new row in a clean csv file
    with open(output_path, "w", encoding= "utf-8", newline="") as outfile:
        writer = csv.writer(outfile)

        # write header
        writer.writerow(["word"])

        for word in word_list:
            writer.writerow([word])

# main function
if __name__ == "__main__":
    clean_nawl_csv(INPUT_PATH, "nawl_cleaned.csv")
    print("NAWL cleaning has been completed.")