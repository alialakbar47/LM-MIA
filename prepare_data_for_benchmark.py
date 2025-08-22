# FILE: prepare_data_for_benchmark.py
import numpy as np
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Convert a guess CSV to a Hugging Face Dataset for the MIA benchmark.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the input guess_*.csv file.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the tokenizer model used to generate the CSV.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the Hugging Face Dataset.")
    args = parser.parse_args()

    # 1. Load the tokenizer used during extraction
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 2. Load your result CSV
    df = pd.read_csv(args.csv_file)

    # 3. Decode the token IDs back into text strings
    try:
        token_ids = [eval(row) for row in df['Suffix Guess']]
    except Exception as e:
        print(f"Error processing 'Suffix Guess' column: {e}")
        return

    text_column = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    # 4. The 'Is Correct' column is your label
    label_column = df['Is Correct'].tolist()

    # 5. Create the Hugging Face Dataset object
    hf_dataset = Dataset.from_dict({
        "text": text_column,
        "label": label_column
    })

    # 6. Save the dataset to the specified directory
    hf_dataset.save_to_disk(args.output_dir)
    print(f"Successfully converted {args.csv_file} to a Hugging Face Dataset at: {args.output_dir}")

if __name__ == "__main__":
    main()