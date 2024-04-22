# COMP3000
Project Scope
This project was originally conceived to encompass a wide range of space mission operations, addressing various aspects of space weather and its impacts. However, due to the intricate and multifaceted nature of these operations, it became clear that a more targeted approach was needed.

Focusing on Solar Wind
The complexity and breadth of space mission operations pose significant challenges, from data collection and processing to modeling and real-time prediction. In order to maintain a manageable scope and ensure depth over breadth in the analysis, this project narrowed its focus to one of the most critical components of space weather: the solar wind.

By concentrating on the solar wind, the project aims to deliver more accurate and robust predictions that are directly applicable to real-world needs. This focused approach allows for a deeper dive into the nuances of solar wind data, resulting in a model that is both practical and scientifically valuable.

The insights gained from this project have applications in planning and operation strategies for current and future space missions, enhancing our ability to protect sensitive equipment and infrastructure from the potentially disruptive effects of space weather.

Solar Wind Forecasting with LSTM Networks
This repository contains the source code and dataset for a solar wind forecasting model using Long Short-Term Memory (LSTM) networks. The project aims to predict solar wind parameters, including solar wind speed and density, using historical data patterns. This model is designed to assist in the mitigation of potential disruptions caused by space weather.

Overview
The solar wind is a stream of charged particles ejected from the Sun's atmosphere. Its interaction with Earth's magnetosphere can lead to various geomagnetic events. Accurate forecasting of solar wind parameters is crucial for planning and mitigating the effects on power grids, communication systems, and satellite operations.

Repository Structure
data/: Contains the datasets used for training and testing the LSTM model, sourced from Kaggle and NOAA.
src/: Source code of the LSTM model including data preprocessing, model training, and evaluation scripts.
models/: Pre-trained models and their weights.
docs/: Additional documentation and references.
requirements.txt: List of dependencies required to run the project.

Getting Started
Prerequisites
Ensure you have the following installed:

Python 3.8 or higher
pip
virtualenv (optional)
Installation
Clone the repository:

git clone [https://github.com/JMY22/COMP3000.git]
cd COMP3000

(Optional) Create and activate a virtual environment:
virtualenv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required dependencies:
pip install -r requirements.txt

Usage
Navigate to the src/ directory and run the main script:
python main.py

For training the model with custom settings, edit the config.json file and then run:
python train.py

To evaluate the model on the test set, run:
python evaluate.py

Datasets
The datasets included in this repository are sourced from Kaggle and NOAA. Ensure to adhere to their respective licenses and usage agreements.

Kaggle Dataset: [link-to-kaggle-dataset]
NOAA Dataset: [link-to-noaa-dataset]

Acknowledgments
NOAA for providing the solar wind data.
Kaggle for hosting the historical dataset.
Contributors and maintainers of the open-source libraries used in this project.
