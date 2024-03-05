import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
import os
import re


def plot(df, title=None, xlabel=None, ylabel=None, sizex=10, sizey=6, legend=False):
    """
    Receives a DataFrame and plots it row by row.
    """

    # Get the header data
    headers = df.columns.values

    plt.figure(figsize=(sizex, sizey))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Rotate the header names for readability
    plt.xticks(rotation=45)

    for index, row in df.iterrows():
        plt.plot(headers, row.values, linewidth=0.5, label=index)
        if legend:
            plt.legend()

    plt.show()


def read_spectrum_file(file_path):
    """
    Reads a CSV file containing a spectrum and converts it to a DataFrame.
    """
    data = pd.read_csv(file_path, header=None)
    data[0] = np.round(data[0], 2).astype(float)
    transposed_df = data.T

    if transposed_df.isna().any().any():
        raise ValueError(f"File {file_path} contains missing data.")

    if transposed_df.shape != (2, 3112):
        raise ValueError(f"File {file_path} does not have the expected shape.")

    transposed_df.columns = transposed_df.iloc[0]
    return transposed_df.iloc[1:].reset_index(drop=True)


def extract_metadata(file_name, pattern, indices):
    """
    Extracts metadata from a file name using a regular expression.
    """
    match = re.search(pattern, file_name)
    if not match:
        raise ValueError(f"Pattern does not match file name: {file_name}")

    treatment_data = {}
    for group, (index_name, convert_to_int) in indices.items():
        group_value = match.group(group)
        if convert_to_int:
            group_value = int(group_value)
        treatment_data[index_name] = group_value

    return treatment_data


def import_spectra(pattern, indices, directory):
    """
    Imports spectra from CSV files in the specified directory.
    """
    spectra = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.CSV'):
            file_path = os.path.join(directory, file_name)
            try:
                spectrum_df = read_spectrum_file(file_path)
                treatment_data = extract_metadata(file_name, pattern, indices)
                spectrum_df = spectrum_df.assign(**treatment_data)
                spectrum_df.set_index(
                    list(treatment_data.keys()), inplace=True)
                spectra.append(spectrum_df)
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
    if spectra:
        return pd.concat(spectra, axis=0, ignore_index=False)
    else:
        return None


def snv(row):
    # Function to apply Standard Normal Variate (SNV) to a row
    mean = np.mean(row)  # Calculate the row mean
    std = np.std(row)    # Calculate the row standard deviation
    if std == 0:         # Avoid division by zero
        return row
    else:
        return (row - mean) / std  # Apply SNV to the row


def apply_savgol(row, window_length, polyorder, deriv):
    smoothed_row = savgol_filter(
        row.values, window_length=window_length, polyorder=polyorder, deriv=deriv)
    return pd.Series(smoothed_row, index=row.index)
