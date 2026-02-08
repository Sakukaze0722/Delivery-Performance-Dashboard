# Delivery Performance Dashboard

A Python Streamlit application for delivery performance analytics using the Brazilian E-Commerce public dataset by Olist.

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── data/
│   ├── raw/            # Raw CSV files (see Dataset section)
│   └── processed/      # Processed fact_orders.parquet (auto-generated)
└── src/
    ├── config.py       # Configuration
    ├── data_load.py    # Data loading
    ├── transform.py    # Data transformation
    ├── metrics.py      # Metrics calculation
    ├── charts.py       # Chart components
    └── styles.py       # Styling utilities
```

## Dataset

**Source:** [Kaggle: Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Place the following CSV files in `data/raw/` (minimum set required):

| File | Description |
|------|-------------|
| `olist_orders_dataset.csv` | Order metadata and status |
| `olist_customers_dataset.csv` | Customer info and location |
| `olist_order_items_dataset.csv` | Order line items |
| `olist_order_payments_dataset.csv` | Payment details |
| `olist_products_dataset.csv` | Product info and category |
| `product_category_name_translation.csv` | Category name translations |
| `olist_geolocation_dataset.csv` | Zip code coordinates |

**Optional (for full dataset):**
- `olist_order_reviews_dataset.csv`
- `olist_sellers_dataset.csv`

### Getting the Data

1. **Option A – Kaggle API (kagglehub):**  
   Ensure Kaggle API credentials are configured. The app will download the dataset on first run.

2. **Option B – Manual download:**  
   Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), extract the archive, and copy the CSV files into `data/raw/`.

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Dashboard Features

- **Filters:** Date range, state, category, payment type, delivered-only
- **KPIs:** Total orders, delivered count, on-time rate, average delay
- **Map:** Delivery delay by region (OpenStreetMap)
- **Histogram:** Distribution of delay days
- **Root causes:** Worst on-time rate by product category
- **Data table:** Filtered fact table with CSV download
