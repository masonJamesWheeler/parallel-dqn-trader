import os
import numpy as np
import requests
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import StringIO, BytesIO
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and Settings
SETTINGS = {
    'start_year': 2000,
    'end_year': datetime.now().year,
    'container_name': 'stock-data',
    'api_base_url': 'https://www.alphavantage.co/query',
    'interval': '15min',
    'requests_per_minute': 75
}


SLEEP_TIME = (60 / SETTINGS['requests_per_minute']) * 4  # 60 seconds divided by max requests per minute

STOCKS = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corp.
    "GOOGL",  # Alphabet Inc.
    "AMZN",  # Amazon.com Inc.
    "META",  # Facebook, Inc.
    "INTC",  # Intel Corp.
    "CSCO",  # Cisco Systems Inc.
    "IBM",  # IBM Corp.
    "ORCL",  # Oracle Corp.
    "NVDA",  # NVIDIA Corp.
    "ADBE",  # Adobe Inc.
    "CRM",  # Salesforce.com Inc.
    "QCOM",  # Qualcomm Inc.
    "AMD",  # AMD Inc.
    "PYPL",  # PayPal Holdings Inc.
    "TSLA",  # Tesla Inc.
    "SHOP",  # Shopify Inc.
    "ZM",  # Zoom Video Communications Inc.
    "SNOW",  # Snowflake Inc.
    "PLTR",  # Palantir Technologies Inc.
    "JPM",  # JPMorgan Chase & Co.
    "BAC",  # Bank of America Corp.
    "WFC",  # Wells Fargo & Co.
    "C",  # Citigroup Inc.
    "GS",  # Goldman Sachs Group Inc.
    "MS",  # Morgan Stanley
    "V",  # Visa Inc.
    "MA",  # Mastercard Inc.
    "AXP",  # American Express Co.
    "BLK",  # BlackRock Inc.
    "BRK.B",  # Berkshire Hathaway Inc.
    "KO",  # Coca-Cola Co.
    "PEP",  # PepsiCo Inc.
    "PG",  # Procter & Gamble Co.
    "PM",  # Philip Morris International Inc.
    "MO",  # Altria Group Inc.
    "JNJ",  # Johnson & Johnson
    "PFE",  # Pfizer Inc.
    "MRK",  # Merck & Co. Inc.
    "ABBV",  # AbbVie Inc.
    "BMY",  # Bristol-Myers Squibb Co.
    "LLY",  # Eli Lilly and Co.
    "AMGN",  # Amgen Inc.
    "GILD",  # Gilead Sciences Inc.
    "UNH",  # UnitedHealth Group Inc.
    "ANTM",  # Anthem Inc.
    "CVS",  # CVS Health Corp.
    "WMT",  # Walmart Inc.
    "TGT",  # Target Corp.
    "COST",  # Costco Wholesale Corp.
    "HD",  # Home Depot Inc.
    "LOW",  # Lowe's Companies Inc.
    "MCD",  # McDonald's Corp.
    "SBUX",  # Starbucks Corp.
    "YUM",  # Yum! Brands Inc.
    "CMG",  # Chipotle Mexican Grill Inc.
    "WEN",  # Wendy's Co.
    "DPZ",  # Domino's Pizza Inc.
    "NFLX",  # Netflix Inc.
    "DIS",  # Walt Disney Co.
    "CMCSA",  # Comcast Corp.
    "T",  # AT&T Inc.
    "VZ",  # Verizon Communications Inc.
    "TMUS",  # T-Mobile US Inc.
    "CHTR",  # Charter Communications Inc.
    "NKE",  # Nike Inc.
    "ADDYY",  # Adidas AG
    "LULU",  # Lululemon Athletica Inc.
    "UAA",  # Under Armour Inc.
    "RL",  # Ralph Lauren Corp.
    "PVH",  # PVH Corp.
    "HBI",  # Hanesbrands Inc.
    "GPS",  # Gap Inc.
    "JWN",  # Nordstrom Inc.
    "ANF",  # Abercrombie & Fitch Co.
    "AEO",  # American Eagle Outfitters Inc.
    "URBN",  # Urban Outfitters Inc.
    "TJX",  # TJX Companies Inc.
    "ROST",  # Ross Stores Inc.
    "KSS",  # Kohl's Corp.
    "M",  # Macy's Inc.
    "DDS",  # Dillard's Inc.
    "TGT",  # Target Corp.
    "BBY",  # Best Buy Co. Inc.
    "GRMN",  # Garmin Ltd.
    "RCL",  # Royal Caribbean Group
    "CCL",  # Carnival Corp.
    "NCLH",  # Norwegian Cruise Line Holdings Ltd.
    "DAL",  # Delta Air Lines Inc.
    "AAL",  # American Airlines Group Inc.
    "LUV",  # Southwest Airlines Co.
    "UAL",  # United Airlines Holdings Inc.
    "BA",  # Boeing Co.
    "LMT",  # Lockheed Martin Corp.
    "NOC",  # Northrop Grumman Corp.
    "RTX",  # Raytheon Technologies Corp.
    "GD",  # General Dynamics Corp.
    "HII",  # Huntington Ingalls Industries Inc.
    "TXT",  # Textron Inc.
    "TSN",  # Tyson Foods Inc.
    "CAG",  # Conagra Brands Inc.
    "HRL",  # Hormel Foods Corp.
    "KHC",  # Kraft Heinz Co.
    "MDLZ",  # Mondelez International Inc.
    "GIS",  # General Mills Inc.
    "CPB",  # Campbell Soup Co.
    "CL",  # Colgate-Palmolive Co.
    "CLX",  # Clorox Co.
    "KMB",  # Kimberly-Clark Corp.
    "EL",  # Estée Lauder Companies Inc.
    "PG",  # Procter & Gamble Co.
    "UL",  # Unilever PLC
    "NSRGY",  # Nestlé S.A.
    "PEP",  # PepsiCo Inc.
    "KO",  # Coca-Cola Co.
    "BUD",  # Anheuser-Busch InBev SA/NV
    "TAP",  # Molson Coors Beverage Co.
    "STZ",  # Constellation Brands Inc.
    "DEO",  # Diageo PLC
    "BF.B",  # Brown-Forman Corp.
    "MO",  # Altria Group Inc.
    "PM",  # Philip Morris International Inc.
    "BTI",  # British American Tobacco PLC
    "IMBBY",  # Imperial Brands PLC
    "VZ",  # Verizon Communications Inc.
    "T",  # AT&T Inc.
    "TMUS",  # T-Mobile US Inc.
    "CMCSA",  # Comcast Corp.
    "CHTR",  # Charter Communications Inc.
    "DIS",  # Walt Disney Co.
    "NFLX",  # Netflix Inc.
    "FOX",  # Fox Corp.
    "VIAC",  # ViacomCBS Inc.
    "TWTR",  # Twitter Inc.
    "SNAP",  # Snap Inc.
    "PINS",  # Pinterest Inc.
    "SPOT",  # Spotify Technology S.A.
    "MTCH",  # Match Group Inc.
    "IAC",  # IAC/InterActiveCorp
    "ETSY",  # Etsy Inc.
    "EBAY",  # eBay Inc.
    "SQ",  # Square Inc.
    "PYPL",  # PayPal Holdings Inc.
    "V",  # Visa Inc.
    "MA",  # Mastercard Inc.
    "AXP",  # American Express Co.
    "DFS",  # Discover Financial Services
    "COF",  # Capital One Financial Corp.
    "SYF",  # Synchrony Financial
    "BK",  # Bank of New York Mellon Corp.
    "STT",  # State Street Corp.
    "SCHW",  # Charles Schwab Corp.
    "ETFC",  # E*TRADE Financial Corp.
    "TD",  # Toronto-Dominion Bank
    "RY",  # Royal Bank of Canada
    "BMO",  # Bank of Montreal
    "BNS",  # Bank of Nova Scotia
    "CM",  # Canadian Imperial Bank of Commerce
    "CNR",  # Canadian National Railway Co.
    "CP",  # Canadian Pacific Railway Ltd.
    "UNP",  # Union Pacific Corp.
    "CSX",  # CSX Corp.
    "NSC",  # Norfolk Southern Corp.
    "KSU",  # Kansas City Southern
    "FDX",  # FedEx Corp.
    "UPS",  # United Parcel Service Inc.
    "RCL",  # Royal Caribbean Group
    "CCL",  # Carnival Corp.
    "NCLH",  # Norwegian Cruise Line Holdings Ltd.
    "DAL",  # Delta Air Lines Inc.
    "AAL",  # American Airlines Group Inc.
    "LUV",  # Southwest Airlines Co.
    "UAL",  # United Airlines Holdings Inc.
    "EXPE",  # Expedia Group Inc.
    "BKNG",  # Booking Holdings Inc.
    "MAR",  # Marriott International Inc.
    "HLT",  # Hilton Worldwide Holdings Inc.
    "IHG",  # InterContinental Hotels Group PLC
    "H",  # Hyatt Hotels Corp.
    "WH",  # Wyndham Hotels & Resorts Inc.
    "VAC",  # Marriott Vacations Worldwide Corp.
    "NKE",  # Nike Inc.
    "LULU",  # Lululemon Athletica Inc.
    "UA",  # Under Armour Inc.
    "COLM",  # Columbia Sportswear Co.
    "VFC",  # VF Corp.
    "GPS",  # Gap Inc.
    "ANF",  # Abercrombie & Fitch Co.
    "URBN",  # Urban Outfitters Inc.
    "TJX",  # TJX Companies Inc.
    "ROST",  # Ross Stores Inc.
    "KSS",  # Kohl's Corp.
    "M",  # Macy's Inc.
    "JWN",  # Nordstrom Inc.
    "DDS",  # Dillard's Inc.
    "LB",  # L Brands Inc.
    "CHWY",  # Chewy Inc.
    "SFIX",  # Stitch Fix Inc.
    "W",  # Wayfair Inc.
    "RH",  # RH (Restoration Hardware)
    "BBY",  # Best Buy Co. Inc.
    "GRMN",  # Garmin Ltd.
    "FIT",  # Fitbit Inc.
    "HOG",  # Harley-Davidson Inc.
    "PII",  # Polaris Inc.
    "F",  # Ford Motor Co.
    "GM",  # General Motors Co.
    "TSLA",  # Tesla Inc.
    "HMC",  # Honda Motor Co. Ltd.
    "TM",  # Toyota Motor Corp.
    "BMWYY",  # Bayerische Motoren Werke AG (BMW)
    "DDAIF",  # Daimler AG
    "VLKAF",  # Volkswagen AG
    "RACE",  # Ferrari N.V.
    "FCAU",  # Fiat Chrysler Automobiles N.V.
    "HOG",  # Harley-Davidson Inc.
    "PII",  # Polaris Inc.
]

# Secure handling of API keys and connection strings
API_KEY = os.environ.get('ALPHAVANTAGE_API_KEY', '')
AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING', '')

if not API_KEY or not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("API_KEY or AZURE_STORAGE_CONNECTION_STRING not set in environment variables")

# Initialize Azure Storage Blob Service
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_intraday_stock_data(symbol, interval='1min', year=None, month=None):
    """
    Fetches intraday stock data for a specific symbol, interval, year, and month.
    """
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'apikey': API_KEY,
        'outputsize': 'full',
        'datatype': 'csv'
    }
    
    if year and month:
        params['month'] = f"{year}-{month:02d}"
    
    try:
        response = requests.get(SETTINGS['api_base_url'], params=params)
        response.raise_for_status()

        time.sleep(SLEEP_TIME)
        
        # Check for API call frequency
        if 'Note' in response.text:
            logger.warning("API call frequency limit reached. Waiting for 60 seconds.")
            time.sleep(60)
            return fetch_intraday_stock_data(symbol, interval, year, month)  # Retry after waiting
        
        # Check if the response contains valid data
        if response.text.strip().startswith('timestamp,open,high,low,close,volume'):
            return response.text
        else:
            logger.error(f"Invalid data received for {symbol} {year}-{month}. Response: {response.text[:200]}...")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data for {symbol} {year}-{month}: {e}")
        return None

def validate_stock_data(data):
    """
    Validates the received stock data.
    """
    if data is None or data.empty or 'timestamp' not in data.columns:
        raise ValueError("Invalid or empty data received")

def save_to_adls(data, symbol, year, month):
    """
    Saves the fetched intraday data to Azure Data Lake Storage in Parquet format.
    """
    if data is None:
        logger.warning(f"No data to save for {symbol} {year}-{month}")
        return

    try:
        df = pd.read_csv(StringIO(data))
        validate_stock_data(df)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group data by date
        grouped = df.groupby(df['timestamp'].dt.date)
        
        for date, group in grouped:
            # Convert DataFrame to Parquet format
            table = pa.Table.from_pandas(group)
            
            # Save to BytesIO buffer
            parquet_buffer = BytesIO()
            pq.write_table(table, parquet_buffer)
            parquet_buffer.seek(0)
            
            # Save to ADLS
            file_name = f'stocks_intraday/{symbol}/{year}/{month:02d}/{symbol}_{date.strftime("%Y-%m-%d")}.parquet'
            blob_client = blob_service_client.get_blob_client(container=SETTINGS['container_name'], blob=file_name)
            blob_client.upload_blob(parquet_buffer.getvalue(), overwrite=True)
            logger.info(f"Intraday data for {symbol} on {date} saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save data to ADLS for {symbol} {year}-{month}: {e}")
        raise

def get_date_range():
    """
    Generates a range of dates from the start year to the current month.
    """
    end_date = datetime.now().replace(day=1) - timedelta(days=1)
    start_date = datetime(SETTINGS['start_year'], 1, 1)
    current_date = start_date
    while current_date <= end_date:
        yield current_date.year, current_date.month
        current_date += relativedelta(months=1)

def process_stock(symbol):
    """
    Processes intraday data for a single stock across all dates.
    """
    for year, month in get_date_range():
        logger.info(f"Fetching intraday data for {symbol} {year}-{month:02d}")
        try:
            data = fetch_intraday_stock_data(symbol, SETTINGS['interval'], year, month)
            save_to_adls(data, symbol, year, month)
        except Exception as e:
            logger.error(f"Error processing {symbol} for {year}-{month}: {e}")

def main():
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_stock, symbol) for symbol in STOCKS]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"An error occurred while processing a stock: {e}")

if __name__ == "__main__":
    main()