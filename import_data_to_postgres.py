import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import numpy as np

def import_excel_to_postgres():
    """
    Import the test-dataset.xlsx into PostgreSQL database
    """
    try:
        # Read the Excel file
        print("Reading Excel file...")
        df = pd.read_excel('upload/test-dataset.xlsx')
        
        # Display basic info about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Clean column names (remove spaces, special characters)
        df.columns = df.columns.str.replace(' ', '_').str.replace(':', '').str.lower()
        
        # Handle missing values - for now, keep them as NULL
        # In a real scenario, you might want to implement more sophisticated imputation
        
        # Create connection to PostgreSQL
        print("Connecting to PostgreSQL...")
        engine = create_engine('postgresql://cardiovascular_user:password123@localhost/cardiovascular_db')
        
        # Import data to PostgreSQL
        print("Importing data to PostgreSQL...")
        df.to_sql('cardiovascular_data', engine, if_exists='replace', index=False, method='multi')
        
        print("Data imported successfully!")
        
        # Display some basic statistics
        print(f"Total records imported: {len(df)}")
        print(f"Columns imported: {len(df.columns)}")
        
        # Test the connection by querying a few records
        print("\nTesting connection - first 5 records:")
        test_query = "SELECT * FROM cardiovascular_data LIMIT 5;"
        result = pd.read_sql(test_query, engine)
        print(result)
        
        return True
        
    except Exception as e:
        print(f"Error importing data: {e}")
        return False

if __name__ == "__main__":
    success = import_excel_to_postgres()
    if success:
        print("\n✅ Data import completed successfully!")
    else:
        print("\n❌ Data import failed!")

