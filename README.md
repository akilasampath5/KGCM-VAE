
# Arctic S2S Multivariate Dataset (2020–2024)

## Dataset Description
This dataset provides daily multivariate oceanographic and sea ice variables for the Arctic region, designed for sub-seasonal to seasonal (S2S) prediction and causal sensitivity analysis.

- **Variables Included:** Sea Surface Height (SSH), sea ice thickness, and surface velocity components (u, v).
- **Temporal Coverage:** 2020–2024 (Total of 1,620 days).
- **Spatial Coverage:** Arctic region [ 60°N to 90°N].
- **Temporal Resolution:** Daily.
## File Structure
The data is provided in a single CSV file: `arctic_s2s_multivar_2020_2024.csv`.

| Column Name | Variable Description | Units |
| :--- | :--- | :--- |
| `uoe` | Eastward sea water velocity | m/s |
| `von` | Northward sea water velocity | m/s |
| `total_vel` | Total surface velocity magnitude | m/s |
| `zos` | Sea surface height (SSH) | m |
| `sithick` | Sea ice thickness | m |

## Data Download
The dataset is hosted on Zenodo and can be accessed via the following DOI:
[https://doi.org/10.5281/zenodo.18273760](https://doi.org/10.5281/zenodo.18273760)
## Example Usage
You can load and inspect the data using Python and the `pandas` library:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('arctic_s2s_multivar_2020_2024.csv')

# Display the first few rows
print(df.head())

print(df.describe())

