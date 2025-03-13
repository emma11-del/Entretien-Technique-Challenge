# Streamlit Application Setup Guide

This guide provides detailed instructions on how to set up and run the Streamlit application (streamlit.py) or the Python script (main.py).

## Prerequisites

Ensure you have Python installed on your system. You can download it from python.org.

## Setting Up the Environment

1. Clone the repository to your local machine or download the source code.

2. Navigate to the project directory:
   cd path/to/project-directory

3. Create a virtual environment if  it's not already create:
   python -m venv venv

4. Activate the virtual environment:
   source venv/bin/activate

5. Install the required packages if it's not already install:
   pip install -r requirements.txt


## Running the Main Script

To run the main.py script, ensure you have the necessary CSV file located at data/intent-detection-train.csv.

Once the CSV is in place, you can execute the script by running:
python scripts/main.py

## Running the Streamlit Application

To run the Streamlit application (streamlit.py), execute the following command in your terminal:
streamlit run scripts/streamlit_app.py


## Deactivating the Virtual Environment

When you are done, you can deactivate the virtual environment by typing:
deactivate

## Troubleshooting

If you encounter any issues related to package installations, ensure your pip is up-to-date:
pip install --upgrade pip

Check if the virtual environment is activated properly and all dependencies are installed as listed in the requirements.txt.

## Additional Information

For more information about Streamlit, visit Streamlit Documentation.
