<p align="center">
  <img src="src/SAInT/dash_application/logo.svg" alt="Logo" style="width: 50%;">
</p>

# SAInT: An Interactive Tool for Sensitivity Analysis In The Loop.

<p align="justify">
We introduce SAInT, an Interactive Tool for Sensitivity Analysis in The Loop, which enables users to train, evaluate, visualize, and explain Machine Learning (ML) models using a graphical interface. Human-in-the-Loop (HITL) tools support informed decision-making through successive iterations with human knowledge. Sensitivity Analysis (SA) is an Explainable Artificial Intelligence (XAI) technique that provides additional insights into model behavior. A key challenge is that using ML typically requires programming skills, especially for integrating XAI methods. This creates barriers to efficient model development and hinders interdisciplinary collaboration. To address this challenge, our tool integrates Interactive ML (IML) with Local SA (LSA) and Global SA (GSA), enabling users to gain insights into model behavior without requiring programming skills. SAInT can be used by Artificial Intelligence (AI) researchers and domain experts. Users can tune hyperparameters, detect data biases, gain insights in model decision-making, select the best features, visualize outliers, apply models on other datasets, explore model reliability, and improve data generation for model refinement.
</p>

## Overview

<p align="center">
  <img src="overview_600dpi.png" alt="Overview" style="width: 80%;">
</p>

<p align="justify">
SAInT integrates Interactive Machine Learning (a-e) and Sensitivity Analysis (f-h) in a loop with user interactions. a) Feature Selection: Define in- and output features. b) Model Configuration: Configure model parameters. c) Model Training: Train or load models. d) Criteria Selection: Select dataset and loss. e) Evaluation: Error plot of all models. The best model is selected. f) Interactive Plot of the best model: Click onto a sample. g) Local Sensitivity Analysis: Local explanations for the selected sample. h) Global Sensitivity Analysis: Identify the best features, which can be used in a) as refinement.
</p>

## Video

[![Watch the video](https://img.youtube.com/vi/m269sWdYUUI/maxresdefault.jpg)](https://www.youtube.com/watch?v=m269sWdYUUI)

If you use our tool, please cite us.

## Installation

We offer two ways of installing our package.

### Fast and easy Installation using pip (recommended)

```
pip install saintool
```

or


### Installation of SAInT using the code of the GitHub Repository

Clone the repository using SSH or HTTP or download and unzip the zip package.

Clone using SSH:
```
git clone git@github.com:dfki-asr/SAInT.git
```
Clone using HTTP:
```
git clone https://github.com/dfki-asr/SAInT.git
```


Go to the main directory.
```
cd SAInT
```
Install the package by executing the script:
```
bash ./install.sh
```
or directly by typing:
```
python3 -m pip install -e .
```
or just
```
pip install -e .
```
## Usage
Start the SAInT Dash application in the browser:
```
python3 -m SAInT
```
The default browser should open automatically.

Alternatively, you can use a browser of your choice by opening a new browser window and type in the address: http://127.0.0.1:8050/
