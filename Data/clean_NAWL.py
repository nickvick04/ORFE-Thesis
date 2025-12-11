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
    '''Function that takes in a file path and outputs'''

    cleaned_rows = []
    # open, read, and store each word in a list
    with open(input_path, "r") as infile:
        for line in infile:
            parts = [
                # strip whitespace and lowercase each word
                word.strip().lower()
                # remove whitespace and newlines, splitting line into words based on commas
                for word in line.strip().split(",")
            ]

            cleaned_rows.append(parts)
    
    # write each word as a new row in a clean csv file
    with open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)

# main function
if __name__ == "__main__":
    clean_nawl_csv(INPUT_PATH, "nawl_cleaned.csv")
    print("NAWL cleaning has been completed.")