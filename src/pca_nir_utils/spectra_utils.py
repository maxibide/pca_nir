import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
import os
import re


def plot(df, title=None, xlabel=None, ylabel=None, sizex=10, sizey=6, legend=False, steps=1000, invert_x=False, log_x=False, xline=False):
    """
    Receives a DataFrame and plots it row by row.
    """

    # Get the header data
    headers = df.columns.values

    fig, ax = plt.subplots(figsize=(sizex, sizey))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Rotate the header names for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xticks(np.arange(min(headers), max(headers)+1, steps))

    if invert_x:
        ax.invert_xaxis()

    if log_x:
        ax.set_xscale('log')

    for index, row in df.iterrows():
        ax.plot(headers, row.values, linewidth=0.5, label=index)
        if legend:
            ax.legend()

    if xline:
        plt.axhline(0, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


def read_spectrum_file(file_path, ignore_shape):
    """
    Reads a CSV file containing a spectrum and converts it to a DataFrame.
    """
    data = pd.read_csv(file_path, header=None)
    data[0] = np.round(data[0], 2).astype(float)
    transposed_df = data.T

    if transposed_df.isna().any().any():
        raise ValueError(f"File {file_path} contains missing data.")

    if not ignore_shape and transposed_df.shape != (2, 3112):
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

def import_spectra(pattern, indices, directory, ignore_shape=False):
    """
    Imports spectra from CSV files in the specified directory.
    """
    spectra = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            try:
                spectrum_df = read_spectrum_file(file_path, ignore_shape)
                treatment_data = extract_metadata(file_name, pattern, indices)
                spectrum_df = spectrum_df.assign(**treatment_data)
                spectrum_df.set_index(
                    list(treatment_data.keys()), inplace=True)
                spectra.append(spectrum_df)
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
            print(f"Processed {file_name}")
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


class Spectra:
    def __init__(self, pattern, indices, directory, ignore_shape=False):
        self.data = import_spectra(pattern, indices, directory, ignore_shape)

    def _copy_with_data(self, new_data):
        new_instance = Spectra.__new__(Spectra)
        new_instance.data = new_data
        return new_instance

    def do_snv(self):
        new_data = self.data.apply(snv, axis=1)
        return self._copy_with_data(new_data)

    def do_savgol(self, window_length, polyorder, deriv):
        new_data = self.data.apply(lambda x: apply_savgol(
            x, window_length, polyorder, deriv), axis=1)
        return self._copy_with_data(new_data)

    def slice(self, min_value, max_value):
        selected_columns = [
            column for column in self.data.columns if min_value <= int(column) <= max_value]
        new_data = self.data[selected_columns]
        return self._copy_with_data(new_data)

    def to_nm(self):
        def wavenumber_to_wavelength(cm):
            return 10000000 / cm

        transformed_columns = [wavenumber_to_wavelength(
            float(column)) for column in self.data.columns]
        new_data = self.data.copy()
        new_data.columns = transformed_columns
        return self._copy_with_data(new_data)

    def filter(self, index, values):
        new_data = self.data.copy()
        filtrados = new_data[new_data.index.get_level_values(index).isin(values)]

        return self._copy_with_data(filtrados)
