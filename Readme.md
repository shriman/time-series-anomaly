# Time-series Sales Forecast - v1

A project to train a sales forecast model using three years of historical daily sales data.

## Project structure

* `data`: contains source data as well as cleaned data

* `models`: path to store trained models

* `notebooks`:
  * `main.ipynb`: notebook to perform data cleaning and initial data exploration

* `src`: source directory for model code as well as Python notebook
  * `ml_core`: contains train and forecast code for corresponding ML models
  * `main.py`: CLI to call train and forecast methods
  
* `requirements.txt` : contains the necessary packages


## Usage
### Setup the requirements:
* `pip install -r ./requirements.txt`

### Data pre-processing
* `main.ipynb`: Python notebook for data analysis, pre-processing and outlier detection

### Model CLI
#### Train
* ` python main.py train --model-type PROPHET`

#### Forecast
* ` python main.py forecast --model-type PROPHET`

#### show help
* ` python main.py --help`


