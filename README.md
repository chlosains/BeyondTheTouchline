# Beyond the Touchline

Welcome to the Beyond the Touchline repository! This repository contains all the code used for football analytics published on my Substack (beyondthetouchline.substack.com). It demonstrates my skills in data analysis, visualisation, and machine learning models using football data.

## **Overview**

This repository showcases:
- Scraping and cleaning football data from sources like Wyscout, FBref and StatsBomb.
- Exploratory data analysis (EDA) of player and team performances.
- Creating visualisations such as xG flow charts, pass maps, and radar charts.
- Machine learning models for player and team performance forecasting.

## **Features**

- **Data Collection**: Scripts to scrape data and convert it into usable formats.
- **Data Analysis**: Python scripts for statistical analysis of football performance data.
- **Visualisations**: Examples of visual content, such as heatmaps, pass networks, and radar charts.
- **Substack Articles**: Links to articles where this code has been used.

## **Getting Started**

### **Dependencies**

Make sure you have the following installed:
- Python 3.8 or higher
- Required Python libraries listed in `requirements.txt`

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/chlosains/BeyondTheTouchline.git

2. Navigate to the project directory:
   ```bash
   cd BeyondTheTouchline

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


### **Usage**

To run the analysis scripts:
1. Choose the script you want to run (e.g., lauren_james_carries.py or corner_logistic_regression_model.py).
2. Make sure you have access to the necessary football data.
3. Run the script using Python:
   ```bash
   python lauren_james_carries.py


## Dataset

The dataset used in the corner_logistic_regression_model project can be downloaded from Figshare:

- [Soccer Match Event Dataset (Figshare)](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2)

### Instructions:

1. Download the following JSON files from the link above:
   - `matches_England.json`
   - `players.json`
   - `events_England.json`
   - `matches_Spain.json`
   - `matches_France.json`
   - `matches_Germany.json`
   - `matches_Italy.json`
   - `events_Spain.json`
   - `events_France.json`
   - `events_Germany.json`
   - `events_Italy.json`

   
2. Save these files in your `Documents` folder or update the paths in the script to point to the location where you store the files.

3. Run the script once the files are downloaded.
