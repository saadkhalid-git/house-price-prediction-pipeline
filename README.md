# House Prices Prediction Pipeline

This project is a machine learning pipeline for predicting house prices. It includes data preprocessing, model training, and model deployment. The pipeline is implemented in Python and uses popular libraries such as pandas, scikit-learn, and joblib.


## Contents

### Data
- `data/processed_df.parquet`: Preprocessed dataset.
- `data/test.csv`: Test dataset.
- `data/train.csv`: Training dataset.

### House Price Prediction Pipeline
- `house_price/build/`: Directory for build artifacts.
- `house_price/dist/`: Directory for distribution artifacts.
- `house_price/house_prices_prediction_pipeline/`: Main package directory.
  - `LICENSE`: License for the project.
  - `pyproject.toml`: Build system requirements and package metadata.
  - `README.md`: Project documentation (this file).
  - `setup.py`: Script for installing the package.

### Models
- `models/categorical_encoder.joblib`: Pretrained categorical encoder.
- `models/model.joblib`: Pretrained model.
- `models/numerical_scaler.joblib`: Pretrained numerical scaler.

### Notebooks
- `notebooks/house-prices-modeling.ipynb`: PW1 Notebook
- `notebooks/model-industrialization-1.ipynb`: PW2 Initial NoteBook
- `notebooks/model-industrialization-final.ipynb`: PW2 final Notebook
- `notebooks/my-1st-notebook.ipynb`: PW0 Notebook

### Other
- `.gitignore`: Git ignore file.
- `requirements.txt`: List of dependencies.

## Installation

To install the project, clone the repository and install the dependencies:

```bash
git clone git@github.com:saadkhalid-git/dsp-saad-khalid.git
cd DSP-SAAD-KHALID
pip install -r requirements.txt
