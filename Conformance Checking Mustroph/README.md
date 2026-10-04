# Predicting Conformance Deviations and Their Positions in Future Event Sequences

This repository contains the implementation of the framework for predicting conformance deviations and their positions in future event sequences using probabilistic suffix prediction. The framework is designed to analyze event logs and predict potential deviations in process executions.

It utilizies the models from probabilistic suffix prediction: https://github.com/ProbabilisticSuffixPredictionLab/Probabilistic_Suffix_Prediction_U-ED-LSTM_pub

## Setting Up the Python Environment with Pipenv

This project uses `pipenv` on Linux devices, for managing Python dependencies. Follow the steps below to set up the virtual environment and install the necessary packages using the provided `Pipfile`.

### Prerequisites
Make sure you have Python and Pipenv installed.
1. **Install Pipenv and pyenv**:
    
    ```bash
    pip install pipenv
    ```

    ```bash
    curl https://pyenv.run | bash
    ```

### Setup Instructions
2. **Navigate to the Project Directory**: Open your terminal and navigate to the project directory where the `Pipfile` is located.
    
    ```bash
    cd path/to/your/project
    ```

3. **Create the Virtual Environment**: Run the following command to create the virtual environment and install the dependencies from the `Pipfile`:
    
    ```bash
    pipenv install
    ```

4. **Activate the Virtual Environment**: Once the environment is set up, activate it with:
    
    ```bash
    pipenv shell
    ```

5. **Data**:
- Download the datasets from the links mentioned in the paper and place them in the `data` folder.
- Adjust all paths in the Jupyter notebooks accordingly.

6. **Proabbilistic Suffix Prediction Model**:
- Clone the Probabilistic Suffix Prediction repository from https://github.com/ProbabilisticSuffixPredictionLab/Probabilistic_Suffix_Prediction_U-ED-LSTM_pub
- Follow the setup instructions in that repository to install any additional dependencies required for the model.

7. **Run the Project**: You can now run the project within the virtual environment. 
- To reproduce the results, execute the Juyter notebooks in `src/notebooks/`.
- To reproduce the results of the reimplemented approaches FFN and LSTM execute the Juyter notebooks in `src/notebooks/reimplemented_approaches/proactive_conformance_checking`.
- To reproduce the results of the original code of BPDP execute the Juyter notebooks in `src/notebooks/reimplemented_approaches/original`. 
