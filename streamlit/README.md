# Streamlit – Nassau Candy Distributor Dashboard

## Project Overview
This folder contains the interactive Streamlit dashboard developed for the Nassau Candy Distributor data analytics project.

The dashboard presents business insights through interactive visualizations, KPIs, filters, and detailed analytical pages.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Plotly

## Dashboard Pages

1. **Executive Overview** – Overall business performance and key KPIs.
2. **Factory Analysis** – Factory-wise sales and profitability analysis.
3. **Product Analysis** – Product performance and ABC classification.
4. **Geography & Routes** – Regional performance and factory-to-customer route combinations.
5. **Shipping Analysis** – Shipping-mode usage and order/ship date analysis.
6. **Transaction Explorer** – Detailed transaction-level data exploration.

## Key Features
- Interactive sidebar navigation
- Dynamic filters
- KPI cards
- Interactive Plotly charts
- Factory and product comparisons
- Geographical analysis
- Detailed transaction exploration

## How to Run the Application

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## Important Data Limitations
The available dataset does not contain verified road distances, GPS coordinates, actual transportation costs, or confirmed delivery performance. Therefore, route analysis is based on factory and destination combinations rather than actual road-route efficiency.

## Project Information
**Project:** Factory-to-Customer Shipping Route Efficiency Analysis  
**Dataset:** Nassau Candy Distributor
