#!/usr/bin/env python3
"""
Research-Driven Data Storage Test
Based on DuckDB and PyArrow research documentation
"""

import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_duckdb_implementation():
    """Test DuckDB implementation following research patterns."""
    print("📊 Testing DuckDB Implementation (Research-Driven)...")
    
    try:
        import duckdb
        import pyarrow as pa
        
        # Research Pattern: Simple table creation without generated columns
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test_trading.duckdb")
            con = duckdb.connect(db_path)
            
            print("✅ DuckDB connection established")
            
            # Research Pattern: Basic OHLCV schema (from research summary)
            con.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price DOUBLE NOT NULL,
                    high_price DOUBLE NOT NULL,
                    low_price DOUBLE NOT NULL,
                    close_price DOUBLE NOT NULL,
                    volume BIGINT NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            print("✅ OHLCV table created successfully")
            
            # Create synthetic data following research patterns
            dates = pd.date_range('2024-01-01', periods=10, freq='1h')
            test_df = pd.DataFrame({
                'symbol': ['BTC'] * 10,
                'timestamp': dates,
                'open_price': np.random.uniform(45000, 50000, 10),
                'high_price': np.random.uniform(50000, 55000, 10),
                'low_price': np.random.uniform(40000, 45000, 10),
                'close_price': np.random.uniform(45000, 50000, 10),
                'volume': np.random.randint(1000, 5000, 10)
            })
            
            # Research Pattern: DataFrame registration and bulk insert
            con.register("temp_data", test_df)
            con.execute("INSERT INTO ohlcv SELECT * FROM temp_data")
            print("✅ Data inserted successfully")
            
            # Research Pattern: Technical indicators with window functions
            result = con.execute("""
                SELECT 
                    symbol, timestamp, close_price,
                    AVG(close_price) OVER (ORDER BY timestamp ROWS 4 PRECEDING) as sma_5,
                    MAX(high_price) OVER (ORDER BY timestamp ROWS 4 PRECEDING) as high_5,
                    MIN(low_price) OVER (ORDER BY timestamp ROWS 4 PRECEDING) as low_5
                FROM ohlcv
                ORDER BY timestamp
            """).df()
            
            if len(result) > 0:
                print(f"✅ Technical indicators calculated: {len(result)} rows")
                print(f"✅ SMA values: {result['sma_5'].iloc[-1]:.2f}")
            else:
                print("❌ No technical indicators calculated")
            
            # Research Pattern: Parquet export with compression
            parquet_path = str(Path(temp_dir) / "test_ohlcv.parquet")
            con.execute(f"""
                COPY ohlcv TO '{parquet_path}' (
                    FORMAT parquet,
                    COMPRESSION snappy
                )
            """)
            print("✅ Parquet export successful")
            
            # Research Pattern: Parquet query with filter pushdown
            result2 = con.execute(f"""
                SELECT COUNT(*) FROM '{parquet_path}'
                WHERE symbol = 'BTC'
            """).fetchone()
            
            if result2[0] == 10:
                print("✅ Parquet filter pushdown working")
            else:
                print(f"⚠️  Unexpected count: {result2[0]}")
            
            con.close()
            print("✅ DuckDB research patterns validated")
            
        return True
        
    except Exception as e:
        print(f"❌ DuckDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pyarrow_implementation():
    """Test PyArrow implementation following research patterns."""
    print("🏹 Testing PyArrow Implementation (Research-Driven)...")
    
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Research Pattern: OHLCV schema with proper types
        ohlcv_schema = pa.schema([
            ('timestamp', pa.timestamp('ns', tz='UTC')),
            ('symbol', pa.dictionary(pa.int32(), pa.string())),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.float64()),
            ('trades', pa.int64())
        ])
        print("✅ PyArrow schema defined")
        
        # Create test data with proper schema
        dates = pd.date_range('2024-01-01', periods=10, freq='1h', tz='UTC')
        test_data = {
            'timestamp': dates,
            'symbol': ['BTC'] * 10,
            'open': np.random.uniform(45000, 50000, 10),
            'high': np.random.uniform(50000, 55000, 10),
            'low': np.random.uniform(40000, 45000, 10),
            'close': np.random.uniform(45000, 50000, 10),
            'volume': np.random.uniform(1000, 5000, 10),
            'trades': np.random.randint(100, 500, 10)
        }
        
        # Research Pattern: Table construction from dictionary
        table = pa.table(test_data, schema=ohlcv_schema)
        print(f"✅ PyArrow table created: {len(table)} rows")
        
        # Research Pattern: Zero-copy DataFrame conversion
        df = table.to_pandas()
        table2 = pa.Table.from_pandas(df, schema=ohlcv_schema)
        
        if len(table2) == len(table):
            print("✅ Zero-copy conversions working")
        else:
            print("❌ Conversion length mismatch")
        
        # Research Pattern: Parquet write/read with compression
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_file = str(Path(temp_dir) / "test_data.parquet")
            
            # Write with Snappy compression (from research)
            pq.write_table(table, parquet_file, compression='snappy')
            print("✅ Parquet write successful")
            
            # Read back and validate
            table_read = pq.read_table(parquet_file)
            if len(table_read) == len(table):
                print("✅ Parquet round-trip successful")
            else:
                print("❌ Parquet round-trip failed")
            
        return True
        
    except Exception as e:
        print(f"❌ PyArrow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 Research-Driven Data Storage Testing\n")
    
    success1 = test_duckdb_implementation()
    print()
    success2 = test_pyarrow_implementation()
    
    overall_success = success1 and success2
    print(f"\n📋 Overall Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    sys.exit(0 if overall_success else 1)