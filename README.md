# NIR Spectra Processing

This repository contains a set of Python tools for processing and analyzing near-infrared (NIR) spectra obtained from a Thermo Scientific Nicolet iS50 FTIR-NIR Spectrophotometer.

## Description

The code in this repository was developed to simplify working with NIR spectra, enabling efficient import, processing, and analysis of large spectral datasets. Some of the key features include:

- Import spectra from CSV files into DataFrames
- Extract metadata from file names using regular expressions.
- Preprocess spectral data, including baseline correction and smoothing.
- Principal Component Analysis (PCA) for dimensionality reduction and visualization of the main sources of variation.
- Clustering using the K-means algorithm and evaluation of cluster quality using metrics such as silhouette score and adjusted Rand score.
- Visualization functions for plotting spectra, PCA results, and clustering results.

## Usage

Here's an example of how to use the code:

```python
import pandas as pd
from pca_object import PCAObject
from spectra_utils import import_spectra

# Import spectra from CSV files
pattern = r'(\d+)_(\d+)_(\d+)'
indices = {'sample': (0, True), 'treatment': (1, True), 'replicate': (2, True)}
directory = 'path/to/csv/files'
spectra_df = import_spectra(pattern, indices, directory)

# Perform PCA and clustering
pca_obj = PCAObject(spectra_df)
loadings_df = pca_obj.get_loadings()
pca_obj.determine_clusters()
pca_obj.cluster(n_clusters=3)
pca_obj.plot_pca('Plot Title', sizex=10, sizey=10, color_index=0, legend_index=0, legend=False, annotated=True, add_clusters=False)
```

## Contributions

Contributions are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
