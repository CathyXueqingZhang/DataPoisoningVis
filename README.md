# Interactive Dashboard for Federated Learning System Analysis

## Introduction
This interactive dashboard is designed to dynamically engage users with visualization data, allowing in-depth exploration of various aspects of federated learning (FL). The dashboard provides insights into data poisoning effects, performance of individual clients, and model learning progression over time in the context of a label flipping attack.

## Features
### Global Model Performance Analysis
The dashboard visualizes the F1 score of the global model under user-specified attack settings to assess performance across the FL learning process. Users can zoom into the F1 score details for both the source victim class and the attack target class over specified epoch rounds.

### Local Model Updates Visualization
Using three-dimensional PCA visualizations, the dashboard presents local model updates from clients during each training round. It identifies malicious and benign clients, allowing users to select specific rounds and clients for detailed analysis of the updates. This feature is inspired by the decoupling effect observed in poisoned and benign updates and helps users understand learning dynamics and pinpoint rounds where performance deterioration due to poisoning occurs.

## Getting Started

### Prerequisites
Ensure you have the following requirements installed:
- Python 3.8 or higher
- All necessary Python packages as listed in `requirements.txt`

### Installation
Clone the repository to your local machine:
```bash
git [repository_url](https://github.com/CathyXueqingZhang/DataPoisoningVis.git)
cd DataPoisoningVis

Install the required packages:
```bash
pip install -r requirements.txt

Running the Application
To deploy the dashboard locally, run the following command:
```bash
python3 app.py
After running the command, navigate to http://127.0.0.1:5000 in your web browser to access the dashboard.
