import numpy as np
import pandas as pd
import openpyxl
from openai import AzureOpenAI
import json
import os
from itertools import combinations, permutations
from typing import List, Dict
import requests
import ast
import random
import sqlite3
from sentence_transformers import SentenceTransformer, util
import warnings
import glob
import time
import shutil

# --- Heuristics and Thresholds ---
TEMPORAL_KEYWORDS = {"year", "month", "week", "day", "quarter", "fiscal"}
GEO_KEYWORDS = {"latitude", "longitude", "geo", "zipcode", "location"}
CARDINALITY_THRESHOLD_PERCENT = 0.25
CARDINALITY_THRESHOLD = 50
DECIMAL_PARSING_THRESHOLD = [1, 0.95, 0.9]
INT_PARSING_THRESHOLD = [1, 0.95, 0.9]
NUMERIC_THRESHOLD = 0.9
DATETIME_PARSING_THRESHOLD = [1, 0.95, 0.9]

# --- Output Column Ordering ---
COLUMN_INFORMATION_OUTPUT_COLUMN_ORDER = [
    'table_name', 'column_name', 'column_understanding_confidence', 'column_description',
    'alternate_column_name', 'provided_data_type', 'inferred_data_type', 'example_values',
    'inferred_unit_of_measurement', 'inferred_role', 'inferred_role_reason', 'data_range_min',
    'data_range_max', 'is_nullable', 'is_unique', 'missing_value_percent', 'cardinality',
    'cardinality_level', 'is_binary', 'is_ordered', 'is_timestamp', 'format'
]

# --- Sentence Transformer Model ---
# This might download the model on first run.
text_encoder_model = SentenceTransformer('all-MiniLM-L6-v2')


# --- Utility Functions ---

def standardize_column_name(df):
    """Converts DataFrame column names to lowercase_with_underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def beautify_column_names(df):
    """Converts DataFrame column names to Title Case With Spaces."""
    df.columns = df.columns.str.replace('_', ' ').str.title()
    return df


# --- Database Interaction Functions ---

def setup_database_from_csvs(db_path, csv_dir, table_names):
    """Creates a SQLite database and populates it with data from CSV files."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    print(f"Created new SQLite database: {db_path}")

    try:
        for table_name in table_names:
            csv_path = os.path.join(csv_dir, f"{table_name}.csv")
            if os.path.exists(csv_path):
                print(f"  - Loading data from {csv_path} into table '{table_name}'...")
                df = pd.read_csv(csv_path)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                print(f"    ...Successfully loaded {len(df)} rows.")
            else:
                print(f"  - WARNING: CSV file not found for table '{table_name}' at: {csv_path}")
        return conn
    except Exception as e:
        print(f"FATAL ERROR during database setup: {e}")
        conn.close()
        return None

def execute_query(conn, query: str) -> pd.DataFrame:
    """Executes a SQL query against the SQLite database and returns a pandas DataFrame."""
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing SQLite query: {e}\nQuery: {query}")
        return pd.DataFrame()

def extract_data(conn, table_name, columns_to_read):
    """Extracts data from a SQLite table."""
    if not isinstance(columns_to_read, list):
        columns_to_read = [columns_to_read]

    columns_str = ', '.join([f'"{col}"' for col in columns_to_read])
    query = f'SELECT {columns_str} FROM "{table_name}"'
    return execute_query(conn, query)

def extract_schema(conn, table_name):
    """Gets column information from a SQLite table using PRAGMA."""
    query = f"PRAGMA table_info('{table_name}')"
    result = execute_query(conn, query)
    if not result.empty:
        result.rename(columns={'name': 'col_name', 'type': 'data_type'}, inplace=True)
        return result[['col_name', 'data_type']]
    return pd.DataFrame()


# --- LLM Interaction Function ---

def call_azure_openai(system_prompt, user_prompt, api_key: str, api_url: str, additional_information=None, temperature: float = 0.0):
    """Calls the Azure OpenAI endpoint with provided messages and returns parsed JSON."""
    headers = {"Content-Type": "application/json", "api-key": api_key}
    if additional_information is not None:
        user_prompt = f"{user_prompt}\n\nHere is the data:\n{json.dumps(additional_information, indent=2, default=str)}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    data = {"messages": messages, "temperature": temperature, "response_format": {"type": "json_object"}}

    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"].strip()
        return json.loads(output)
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return {"error": str(e)}

# --- STAGE 1: DESCRIPTION FILLING LOGIC ---

def get_column_value_summary_for_desc(conn, table_name, column_metadata_df):
    """Simplified summary function to get context for description generation."""
    results = []
    for col in column_metadata_df['col_name'].values:
        df = extract_data(conn, table_name, [col])
        if df.empty: continue
        col_data = df[col]
        unique_vals = col_data.nunique(dropna=True)
        results.append({
            "column_name": col,
            "example_values": str(random.sample(col_data.dropna().unique().tolist(), min(5, unique_vals))),
            "cardinality": unique_vals,
        })
    return pd.DataFrame(results)

def infer_column_datatype_role_for_desc(conn, table_name, column_metadata_df):
    """Simplified role inference to get context for description generation."""
    results = []
    for col in column_metadata_df['col_name'].values:
        df = extract_data(conn, table_name, [col])
        if df.empty: continue
        col_data = df[col]
        
        inferred_dtype = "Unknown"
        if pd.api.types.is_numeric_dtype(col_data):
            inferred_dtype = 'integer' if all(col_data.dropna() % 1 == 0) else 'decimal'
        elif pd.api.types.is_string_dtype(col_data):
            inferred_dtype = 'string'
            
        role = "Dimension"
        if inferred_dtype in ['integer', 'decimal'] and col_data.nunique() > CARDINALITY_THRESHOLD:
            role = "Fact"
        
        results.append({
            "column_name": col,
            "inferred_data_type": inferred_dtype,
            "inferred_role": role,
        })
    return pd.DataFrame(results)

def fill_missing_descriptions(conn, input_excel_path, output_excel_path, api_key, api_url, warehouse_info):
    """
    Reads an input data dictionary, generates missing descriptions using an LLM,
    and saves a new, completed data dictionary.
    """
    print("\n--- Starting: Description Filling Stage ---")
    
    try:
        all_sheets = pd.read_excel(input_excel_path, sheet_name=None)
        table_info_df = standardize_column_name(all_sheets['Table Information'])
        column_info_df = standardize_column_name(all_sheets['Column Information'])
    except (FileNotFoundError, KeyError) as e:
        print(f"FATAL ERROR: Could not load the initial data dictionary. Details: {e}")
        return False
        
    # *** MODIFICATION START ***
    # Get the DW description to use in prompts
    dw_description = warehouse_info.get('description', 'A general data warehouse.')
    print(f"  - Using Data Warehouse Context: '{dw_description}'")

    # --- Generate Technical Context ---
    all_generated_column_info = []
    for table_name in table_info_df['table_name'].unique():
        schema_df = extract_schema(conn, table_name)
        if schema_df.empty: continue
        
        summary_df = get_column_value_summary_for_desc(conn, table_name, schema_df)
        role_df = infer_column_datatype_role_for_desc(conn, table_name, schema_df)
        
        merged_df = pd.merge(schema_df.rename(columns={'col_name': 'column_name'}), summary_df, on='column_name')
        merged_df = pd.merge(merged_df, role_df, on='column_name')
        merged_df['table_name'] = table_name
        all_generated_column_info.append(merged_df)

    if not all_generated_column_info:
        print("FATAL: Could not generate any technical metadata for description filling.")
        return False
        
    generated_column_df = pd.concat(all_generated_column_info, ignore_index=True)

    # Merge generated context with user input
    final_column_df = pd.merge(
        generated_column_df,
        column_info_df[['table_name', 'column_name', 'column_description']],
        on=['table_name', 'column_name'],
        how='left'
    )

    # --- Use LLM to Fill Missing Descriptions ---
    SYSTEM_PROMPT = "You are an expert data analyst specializing in creating clear, concise data documentation."
    
    # Updated prompt for columns to include DW description
    PROMPT_FOR_COLUMNS = """
    The data comes from a data warehouse described as: '{dw_description}'

    Your task is to write a concise, one-sentence business description for each database column provided.
    Use the overall data warehouse context, the table's name, its description, and all column metadata to infer the column's business purpose.
    You MUST return a JSON object with a list named 'descriptions'. Each object in the list needs 'column_name' and 'description'.
    """
    
    # Updated prompt for tables to include DW description
    PROMPT_FOR_TABLE = """
    The data comes from a data warehouse described as: '{dw_description}'

    Your task is to write a comprehensive, 2-3 sentence business description for the database table.
    Synthesize the overall data warehouse context, the table name, and the list of its columns (with their business descriptions) to explain the table's purpose and role within the warehouse.
    You MUST return a JSON object with a single key: 'table_description'.
    """
    
    for table_name in table_info_df['table_name'].unique():
        print(f"  Enriching table: '{table_name}'")
        
        # Part A: Generate Column Descriptions
        cols_for_table = final_column_df[final_column_df['table_name'] == table_name]
        cols_to_process = cols_for_table[pd.isna(cols_for_table['column_description']) | (cols_for_table['column_description'] == '')]

        if not cols_to_process.empty:
            print(f"    - Found {len(cols_to_process)} columns needing descriptions.")
            payload = {
                "table_name": table_name,
                "table_description": table_info_df.loc[table_info_df['table_name'] == table_name, 'table_description'].iloc[0] or "Not provided.",
                "columns_to_describe": cols_to_process[['column_name', 'inferred_data_type', 'example_values', 'inferred_role', 'cardinality']].to_dict('records')
            }
            # Format the prompt with the DW description before sending
            user_prompt_for_cols = PROMPT_FOR_COLUMNS.format(dw_description=dw_description)
            ai_response = call_azure_openai(SYSTEM_PROMPT, user_prompt_for_cols, api_key, api_url, payload)
            
            if ai_response and 'descriptions' in ai_response:
                for item in ai_response['descriptions']:
                    mask = (final_column_df['table_name'] == table_name) & (final_column_df['column_name'] == item['column_name'])
                    final_column_df.loc[mask, 'column_description'] = item['description']
                print(f"    - Generated {len(ai_response['descriptions'])} column descriptions.")

        # Part B: Generate Table Description
        table_row_index = table_info_df[table_info_df['table_name'] == table_name].index[0]
        current_table_desc = table_info_df.loc[table_row_index, 'table_description']
        
        if pd.isna(current_table_desc) or current_table_desc == '':
            print("    - Table description is missing. Generating...")
            updated_cols = final_column_df[final_column_df['table_name'] == table_name]
            payload = {"table_name": table_name, "columns": updated_cols[['column_name', 'column_description']].to_dict('records')}
            # Format the prompt with the DW description before sending
            user_prompt_for_table = PROMPT_FOR_TABLE.format(dw_description=dw_description)
            ai_response = call_azure_openai(SYSTEM_PROMPT, user_prompt_for_table, api_key, api_url, payload)
            
            if ai_response and 'table_description' in ai_response:
                table_info_df.loc[table_row_index, 'table_description'] = ai_response['table_description']
                print("    - Generated table description.")
    # *** MODIFICATION END ***
                
    # --- Save the Enriched File ---
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        beautify_column_names(table_info_df.copy()).to_excel(writer, sheet_name='Table Information', index=False)
        # Use the final_column_df which has the descriptions filled
        final_cols_to_write = final_column_df[['table_name', 'column_name', 'column_description']]
        beautify_column_names(final_cols_to_write.copy()).to_excel(writer, sheet_name='Column Information', index=False)
        # Add original DW Info sheet
        dw_info_df = standardize_column_name(pd.read_excel(input_excel_path, sheet_name='DW Information'))
        beautify_column_names(dw_info_df.copy()).to_excel(writer, sheet_name='DW Information', index=False)

    print(f"--- Description Filling Complete. Enriched dictionary saved to '{output_excel_path}' ---\n")
    return True

# --- STAGE 2: KNOWLEDGE LAYER GENERATION LOGIC (from notebook) ---
# Note: Many functions are already defined above. I will only add the ones specific to this stage.

def get_column_value_summary(conn, table_name: str, column_metadata_df):
    """Calculates detailed statistics for each column."""
    results = []
    for col in column_metadata_df['col_name'].values:
        df = extract_data(conn, table_name, [col])
        if df.empty: continue
        col_data = df[col]
        
        unique_vals = col_data.nunique(dropna=True)
        total_vals = len(col_data)
        missing_vals = col_data.isnull().sum()
        
        unique_value_list = col_data.dropna().unique().tolist()
        sample_values = unique_value_list if unique_vals <= 20 else random.sample(unique_value_list, 20)

        results.append({
            "table_name": table_name, "column_name": col,
            "missing_values": missing_vals, "unique_values": unique_vals,
            "total_values": df.shape[0], "is_nullable": str(missing_vals > 0),
            "is_unique": str(unique_vals == total_vals if total_vals > 0 else False),
            "cardinality": unique_vals,
            "cardinality_level": 'High' if unique_vals >= CARDINALITY_THRESHOLD else 'Low',
            "is_binary": str(unique_vals == 2),
            "is_ordered": str(pd.api.types.is_numeric_dtype(col_data) and (col_data.is_monotonic_increasing or col_data.is_monotonic_decreasing)),
            "example_values": sample_values
        })
    return pd.DataFrame(results)

def infer_column_datatype_using_llm(column_metadata_df, column_descriptions, sample_table_data, api_key, api_url):
    """Uses LLM to infer data types based on samples and metadata."""
    print('Using LLM to infer datatype...')
    column_metadata_with_description = pd.merge(left=column_descriptions, right=column_metadata_df, left_on='column_name', right_on='col_name', how='right')
    column_metadata_with_description = column_metadata_with_description[['col_name', 'data_type', 'column_description']]
    column_metadata_with_description.rename(columns={'col_name': 'column_name'}, inplace=True)
    column_description_datatype_sample_data_df = pd.merge(left=column_metadata_with_description, right=sample_table_data, on='column_name', how='left')
    column_description_datatype_sample_data = column_description_datatype_sample_data_df.to_dict(orient='records')
    
    SYSTEM_PROMPT = "You are a data analysis assistant. Your job is to infer the datatype of table columns using their name, description, provided datatype as in database, and up to 100 sample values from the column."
    USER_PROMPT =  """
Infer the most appropriate column datatype using the following options only:
- string
- numeric
- datetime
- boolean
- NA (if datatype is unclear or inconsistent)

If the inferred datatype is `datetime`, also return the format in which the values are stored (e.g., `YYYY-MM-DD`, `YYYYMM`, `DD/MM/YYYY`, or UNIX timestamp).

Follow these rules strictly:
- If **even a single value** in the sample is a string, the column should be inferred as `string`.
- If **all values** are datetime-like (or stored as **integers with recognizable date patterns** like `YYYYMM` or 'MM' or 'YYYY' etc), infer as `datetime` and extract the format.
- If **all values** are numeric and consistent with the description and name, infer as `numeric`.
- If values are **only** `True/False`, `Yes/No`, `0/1`, or similar, infer as `boolean`.
- If values are missing, inconsistent, or not enough to determine, mark as `NA`.

Your response must be a list of objects. Each object must contain:
- `column_name`
- 'provided_data_type'
- `llm_inferred_column_datatype`
- `format` (set to `NA` if not applicable)

Your goal is to infer how the values in each column are stored based on realistic interpretation.

Here is the column metadata from a table. For each column, please infer the datatype based on the column name, description, provided datatype, and sample values. Return your result as a list of dictionaries, as described above.

Return a JSON list of objects. Format:

[
  {{
    "column_name": "...",
    "provided_data_type": "...",
    "llm_inferred_column_datatype": "...",
    "format": "..."
  }},
  ...
]
"""
    
    llm_inferred_datatype = call_azure_openai(SYSTEM_PROMPT, USER_PROMPT, api_key, api_url, additional_information=column_description_datatype_sample_data)

    if isinstance(llm_inferred_datatype, dict) and 'error' in llm_inferred_datatype:
        return pd.DataFrame() 
    data_list = llm_inferred_datatype if isinstance(llm_inferred_datatype, list) else llm_inferred_datatype.get('result', [])
        
    return pd.DataFrame(data_list)

def infer_column_datatype_role(conn, table_name, column_metadata_df, column_descriptions, sample_table_data, api_key, api_url, cardinality_threshold_percent=CARDINALITY_THRESHOLD_PERCENT,
    cardinality_threshold=CARDINALITY_THRESHOLD,
    decimal_threshold=DECIMAL_PARSING_THRESHOLD,
    int_threshold=INT_PARSING_THRESHOLD,
    numeric_threshold=NUMERIC_THRESHOLD,
    datetime_threshold=DATETIME_PARSING_THRESHOLD,
    temporal_keywords=TEMPORAL_KEYWORDS,
    geo_keywords=GEO_KEYWORDS):
    """
    Identify each column as either a Dimension or Fact based on data type, cardinality, and naming heuristics.
    """
    results = []
    llm_inferred_datatype = infer_column_datatype_using_llm(
        column_metadata_df, column_descriptions, sample_table_data, api_key, api_url
    )

    for col in column_metadata_df['col_name'].values:
        print(f'Inferring datatype and role for column {col}')
        df = extract_data(conn, table_name, [col])
        col_data = df[col]
        col_dtype = column_metadata_df.loc[column_metadata_df['col_name'] == col, 'data_type'].values[0]
        llm_dtype = llm_inferred_datatype.loc[llm_inferred_datatype['column_name'] == col, 'llm_inferred_column_datatype'].values[0]
        print(f'Provided Datatype is {col_dtype} and LLM inferred Datatype is {llm_dtype}')
        unique_vals = col_data.nunique(dropna=True)
        total_vals = len(col_data)
        missing_vals = col_data.isnull().sum() 

        col_lc = col.lower()
        inferred_dtype = col_dtype
        datatype_conversion_confidence = 'High'
        is_datetime_like = False
        is_number_like = False
        fmt = llm_inferred_datatype.loc[llm_inferred_datatype['column_name'] == col, 'format'].values[0]

        role = "Dimension"  # Default
        reason = []

        if missing_vals == total_vals:
            role = "Unknown"
            reason.append("All values missing")
        else:
            if (col_dtype.lower() in ['text', 'varchar', 'char', 'character varying']) & (llm_dtype.lower().startswith(('date', 'time'))):
                print('check if datetime')
                converted = pd.to_datetime(col_data, errors='coerce')
                parsed_count = converted.notnull().sum()
                not_null_count = col_data.notnull().sum()
                if ((parsed_count / not_null_count) >= datetime_threshold[0]):
                    is_datetime_like = True
                    inferred_dtype = "datetime"
                    col_data = converted
                    datatype_conversion_confidence = 'High'
                elif ((parsed_count / not_null_count) >= datetime_threshold[1]):
                    is_datetime_like = True
                    inferred_dtype = "datetime"
                    col_data = converted
                    datatype_conversion_confidence = 'Medium'
                elif ((parsed_count / not_null_count) >= datetime_threshold[2]):
                    is_datetime_like = True
                    inferred_dtype = "datetime"
                    col_data = converted
                    datatype_conversion_confidence = 'Low'
                else:
                    is_datetime_like = False
                    inferred_dtype = "string"

            if (col_dtype.lower() in ['text', 'varchar', 'char', 'character varying']) & (llm_dtype.lower().startswith(('numeric'))):
                print('check if number')
                # Try numeric conversion
                converted_numeric = pd.to_numeric(col_data, errors='coerce')
                parsed_count = converted_numeric.notnull().sum()
                not_null_count = col_data.notnull().sum()
                if ((parsed_count / not_null_count) >= numeric_threshold):
                    print('check if number')
                    is_number_like = True
                    inferred_dtype = "decimal"
                    try:
                        converted_int = col_data.astype('int64', errors='ignore')
                        parsed_count_int = converted_int.notnull().sum()
                        not_null_count_int = col_data.notnull().sum()
                        if ((parsed_count_int / not_null_count_int) >= int_threshold[0]):
                            inferred_dtype = "integer"
                            col_data = converted_int
                            datatype_conversion_confidence = 'High'
                        else:
                            if ((parsed_count / not_null_count) >= decimal_threshold[0]):
                                col_data = converted_numeric
                                datatype_conversion_confidence = 'High'
                            elif ((parsed_count / not_null_count) >= decimal_threshold[1]):
                                datatype_conversion_confidence = 'Medium'
                            elif ((parsed_count / not_null_count) >= decimal_threshold[2]):
                                datatype_conversion_confidence = 'Low'
                    except:
                        col_data = converted_numeric
                        datatype_conversion_confidence = 'High'

            if llm_dtype.lower().startswith(('date', 'time')):   # scenario when datetime is stored as int (like MMYYYY)
                is_datetime_like = True
                inferred_dtype = "datetime"
            elif col_dtype.lower() in ['integer', 'bigint', 'smallint', 'int2', 'int4', 'int8']:
                inferred_dtype = "integer"
                is_number_like = True
                datatype_conversion_confidence = 'High'
            elif col_dtype.lower() in ['decimal', 'numeric', 'double precision', 'real', 'float4', 'float8']:
                inferred_dtype = "decimal"
                is_number_like = True
                datatype_conversion_confidence = 'High'
            elif col_dtype.lower() in ['date', 'time', 'timestamp', 'timestamptz', 'timetz']:
                inferred_dtype = "datetime"
                is_datetime_like = True
            elif llm_dtype.lower().startswith('bool') or col_dtype.lower() == 'boolean':
                inferred_dtype = "boolean"
            else:
                inferred_dtype = col_dtype
                is_datetime_like = False
                is_number_like = False
                datatype_conversion_confidence = 'High'

            print('checking fact or dimension')
            if any(keyword in col.lower() for keyword in geo_keywords):
                role = "Dimension"
                reason.append("Geospatial column (name-based)")

            # Heuristic: Numeric columns with high cardinality are facts
            elif pd.api.types.is_numeric_dtype(col_data):
                if ((unique_vals / total_vals) > cardinality_threshold_percent) or (unique_vals > cardinality_threshold):
                    role = "Fact"
                    reason.append("High-cardinality numeric field")
                else:
                    role = "Dimension"
                    reason.append("Low-cardinality numeric field")

            elif any(keyword in col.lower() for keyword in temporal_keywords):
                role = "Dimension"
                reason.append("Temporal column (name-based)")

            elif is_datetime_like or pd.api.types.is_datetime64_any_dtype(col_data):
                role = "Dimension"
                reason.append("Datetime field or parsable string")

            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.infer_dtype(col_data) == "string":
                if ((unique_vals / total_vals) > cardinality_threshold_percent) or (unique_vals > cardinality_threshold):
                    role = "Dimension"
                    reason.append("High-cardinality Categorical string field")
                else:
                    role = "Dimension"
                    reason.append("Low cardinality Categorical string field")
            else:
                role = "Unknown"
                reason.append("Unknown Data Type Field")

            # Adding code for range
            try:
                if inferred_dtype.lower() in ['integer', 'decimal', 'datetime', 'date', 'timestamp', 'numeric', 'double precision', 'real', 'bigint', 'smallint', 'int2', 'int4', 'int8', 'float4', 'float8']:
                    data_range_min, data_range_max = col_data.min(), col_data.max()
                else:
                    data_range_min, data_range_max = None, None
            except:
                data_range_min, data_range_max = None, None

        fmt = fmt if is_datetime_like else None

        # Convert datetime range min/max to string if inferred_dtype is datetime
        if inferred_dtype.lower() == "datetime":
            if data_range_min is not None:
                data_range_min = str(data_range_min)
            if data_range_max is not None:
                data_range_max = str(data_range_max)

        results.append({
            "table_name": table_name,
            "column_name": col,
            "provided_data_type": col_dtype,
            "inferred_data_type": inferred_dtype,
            'datatype_conversion_confidence': datatype_conversion_confidence,
            "data_range_min": data_range_min,
            "data_range_max": data_range_max,
            "inferred_role": role,
            "inferred_role_reason": "; ".join(reason),
            "is_timestamp": is_datetime_like,
            "format": fmt,
        })

    return pd.DataFrame(results)
def get_column_metadata_from_llm(table_name, table_description, column_metadata_with_description, api_key, api_url):
    """Enriches column metadata with synonyms and units of measurement."""
    print('Enriching column metadata with LLM...')
    payload = {
        'table_description': table_description,
        'column_information': column_metadata_with_description.to_dict(orient='records')
    }
    
    SYSTEM_PROMPT = "You are a metadata enrichment assistant that can infer units of measurement and synonyms based on column names and descriptions."
    USER_PROMPT = f"""
Given the following list of column metadata (table name, column name, column_description, and sample values),
generate the following for each column:
- inferred_unit_of_measurement (e.g., Dollar, Millions of Dollar, Percentage, Fraction, Count, Average, Name, City, etc.).
- alternate_column_name: List of 3â€"5 appropriate business names for columns based on column_description, column_name and table_name. These names should mimic what business user may call this column in general.
- generation_confidence: Confidence with which the value for data_unit and synonyms were generated. Accepted values "High", "Medium", "Low".

- If unit is not clear from description, return 'NA'

Return a JSON list of objects. Format:

[
  {{
    "column_name": "...",
    "table_name": "...",
    "inferred_unit_of_measurement": "...",
    "alternate_column_name": ["...", "..."],
    "generation_confidence: "..."
  }},
  ...
]
"""
    
    llm_response = call_azure_openai(SYSTEM_PROMPT, USER_PROMPT, api_key, api_url, additional_information=payload)
    
    data_list = []
    if isinstance(llm_response, dict) and 'result' in llm_response:
        data_list = llm_response['result']
    elif isinstance(llm_response, list):
        data_list = llm_response

    if not data_list:
        return column_metadata_with_description

    df_generated = pd.DataFrame(data_list)
    return pd.merge(column_metadata_with_description, df_generated, on=['table_name', 'column_name'], how='left')

def derive_overall_column_confidence(column_information):
    def get_confidence(row):
        desc_len = len(row['column_description']) if pd.notnull(row['column_description']) else 0
        dtype_conf = row['datatype_conversion_confidence']
        gen_conf = row['generation_confidence']

        high_count = sum([conf == 'High' for conf in [dtype_conf, gen_conf]])
        medium_count = sum([conf == 'Medium' for conf in [dtype_conf, gen_conf]])
        low_count = sum([conf == 'Low' for conf in [dtype_conf, gen_conf]])

        if desc_len >= 10:
            if low_count >= 1:
                return 'Low'
            elif medium_count >= 1 and low_count == 0:
                return 'Medium'
            else:
                return 'High'
        else:
            if low_count >= 1:
                return 'Low'
            else:
                return 'Medium'

    column_information['column_understanding_confidence'] = column_information.apply(get_confidence, axis=1)
    column_information.drop(columns=['datatype_conversion_confidence', 'generation_confidence'], inplace=True)
    return column_information

def get_pk_candidate(conn, table_name, columns_for_pk_sorted: list) -> list:
    """Finds primary key candidates by checking for uniqueness."""
    if not columns_for_pk_sorted: return []
    
    row_count_df = execute_query(conn, f'SELECT COUNT(*) AS row_count FROM "{table_name}"')
    if row_count_df.empty or row_count_df.iloc[0]['row_count'] == 0: return []
    row_count = row_count_df.iloc[0]['row_count']

    for i in range(1, len(columns_for_pk_sorted) + 1):
        for combo in combinations(columns_for_pk_sorted, i):
            cols = list(combo)
            cols_str = ', '.join([f'"{c}"' for c in cols])
            unique_count_query = f'SELECT COUNT(*) as unique_count FROM (SELECT DISTINCT {cols_str} FROM "{table_name}")'
            unique_count_df = execute_query(conn, unique_count_query)
            
            if not unique_count_df.empty and unique_count_df.iloc[0]['unique_count'] == row_count:
                print(f"Found PK candidate: {cols}")
                return [cols] # Return the first and shortest one found
    return []

def get_primary_key(conn, table_name, table_description, column_metadata_df):
    """Identifies primary key candidates."""
    # Simplified exclusion logic
    excluded_mask = (
        (column_metadata_df.missing_value_percent >= 1) |
        (column_metadata_df.cardinality_level == 'Low') |
        (column_metadata_df.inferred_data_type.str.contains('date|time', case=False)) |
        (column_metadata_df.inferred_data_type.str.contains('decimal|numeric|double|real|boolean', case=False))
    )
    
    columns_included = column_metadata_df[~excluded_mask]
    
    sorted_candidates = columns_included.sort_values(by='cardinality', ascending=False)['column_name'].tolist()
    
    return get_pk_candidate(conn, table_name, sorted_candidates)

def enrich_table_information_with_llm(table_name, table_description, column_metadata, api_key, api_url):
    """Gets table-level synonyms and fact/dimension classification from LLM."""
    payload = {
        "table_name": table_name,
        "table_description": table_description,
        "column_metadata": column_metadata.to_dict(orient='records')
    }
    SYSTEM_PROMPT = f"""
    You are an expert Data Engineer and DBA who interacts daily with business users.
    """
    USER_PROMPT = f"""
Given the following: table_name, table_description, list of column metadata (table name, column name, column description, datatype, count of missing_values, cadinality indicating number of unique values, and sample values),
generate the following for the entire table as single output:
- table_name_synonyms: should be a list of upto 5 business names for the table. These names should indicate what a business users calls a table in the domain/industry for which data belongs.
- fact_or_dimension: If the table is a fact table return 'Fact' else if table is dimension table return 'Dimension'. Else return NA.
Also, return your reasoning behind your decision.

Return a JSON list of objects. Format:

[
  {{
    "table_name": "...",
    "table_name_synonyms": ["...", "..."]
    "fact_or_dimension": "..."
  }},
  ...
]
"""
    response = call_azure_openai(SYSTEM_PROMPT, USER_PROMPT, api_key, api_url, additional_information=payload)
    
    default = {"table_name_synonyms": [], "fact_or_dimension": "NA"}
    if isinstance(response, list) and response:
        return response[0]
    elif isinstance(response, dict) and 'error' not in response:
        return response
    return default

def get_table_summary_from_llm(table_name, table_description, column_metadata, sample_data, api_key, api_url):
    """Generates a comprehensive summary of the table for analysts."""
    payload = {
        'table_name': table_name, 'table_description': table_description,
        'column_metadata': column_metadata.to_dict(orient='records'),
        'table_data_json': sample_data.to_dict(orient='records')
    }
    SYSTEM_PROMPT = '''You are an expert Data Engineer and Data Analyst with deep knowledge of data modeling, warehousing, and business intelligence.
You write documentation that helps analysts understand data assets, their structure, purpose, and how they should be queried.
'''
    USER_PROMPT = f"""
You are given metadata (and optionally a few data samples) from a table in a data warehouse.

Your task is to generate a **comprehensive and helpful summary** of the table for use by data analysts.

Use the following metadata:
- Table name and optional table description
- A list of columns with:
    - column name
    - column description
    - data type
    - cardinality (number of distinct values)
    - sample values (if cardinality < 100, all unique values else top 100 unique values)
    - nullable or not
    - inferred datatype
    - unit of measurement (e.g. ID, measure, timestamp, text, geographic field, count, percentage etc.)

If data samples are provided, use them to refine your understanding of the table semantics. Note that data samples doesn't contain all column. Categorical columns with cardinality <10 are ignored.

Your output summary should include:
- **Table Purpose**: What does this table represent? How might it be used?
- **Table Type**:
  - Fact or Dimension?
  - If Fact: what kind (transactional, snapshot, accumulating snapshot, periodic snapshot)?
  - If Dimension: what kind (conformed, junk, degenerate, slowly changing, etc.)?
    - If Dimension thatn if it is SCD type 2 or 3. 

- **How data is updated**: Is this likely append-only? Updated in-place? Slowly changing?
- **Primary key or identifier fields**
- **Temporal fields and time behavior** (e.g., latest records, historical tracking)
- **Geographic or organizational context** (if applicable)
- **Relationships (if inferable)** to other known dimensions or facts
- **Important metrics or measures**
- **Usage guidance**: Any advice for querying? Filters that are typically used? Pitfalls?

Points to note when generating response
- You must not include information from table_description in your output. use it only for reference purpose.
- The response must not be vague or generic. If information is not available for any section, then leave it blank.
- The response should be written in simple English for easy understanding

Return your output as a well-written text (inside a json object) that can be directly pasted into a data documentation tool.

Return a JSON list of objects. Format:

[
  {{
    "table_name": "...",
    "table_purpose": "...",
    "table_type": "...",
    "how_data_is_updated": "...",
    "primary_key_or_identifier_fields": "...",
    "temporal_fields_and_time_behavior": "...",
    "geographic_or_organizational_context": "...",
    "relationships_if_inferable": "...",
    "important_metrics_or_measures": "...",
    "usage_guidance": "...",
    "pitfalls": "..."
  }},
]
"""
    response = call_azure_openai(SYSTEM_PROMPT, USER_PROMPT, api_key, api_url, additional_information=payload)
    
    default = {k: "Not generated." for k in ["table_purpose", "table_type", "how_data_is_updated", "primary_key_or_identifier_fields", "important_metrics_or_measures", "usage_guidance"]}
    if isinstance(response, list) and response:
        return response[0]
    elif isinstance(response, dict) and 'error' not in response:
        return response
    return default
    
def compute_similarity(text1, text2):
    """Computes cosine similarity between two text strings."""
    embeddings = text_encoder_model.encode([str(text1), str(text2)], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return float(similarity)

def get_fk_match_percent(conn, left_table_name, left_column_name, right_table_name, right_column_name):
    """Calculates the percentage of values in the right column that exist in the left."""
    query = f"""
    WITH right_keys AS (SELECT DISTINCT "{right_column_name}" AS key FROM "{right_table_name}" WHERE "{right_column_name}" IS NOT NULL),
         left_keys AS (SELECT DISTINCT "{left_column_name}" AS key FROM "{left_table_name}" WHERE "{left_column_name}" IS NOT NULL)
    SELECT
      (SELECT COUNT(*) FROM right_keys) AS total_right_keys,
      (SELECT COUNT(*) FROM right_keys rk INNER JOIN left_keys lk ON rk.key = lk.key) AS matched_keys_count
    """
    result = execute_query(conn, query)
    if result.empty: return 0.0

    total_right = result.iloc[0]['total_right_keys']
    matched = result.iloc[0]['matched_keys_count']
    
    return round((matched / total_right), 2) if total_right > 0 else 0.0

def foreign_key_search(conn, left_table_name, right_table_name, table_column_metadata):
    """Heuristically searches for foreign key relationships between two tables."""
    # Thresholds
    COL_DESC_MATCH_THRESHOLD = 0.5
    FK_MATCH_THRESHOLDS = {'name': 0.8, 'desc': 0.7, 'value': 0.9}
    
    relationships = []
    left_cols = table_column_metadata[table_column_metadata['table_name'] == left_table_name]
    right_cols = table_column_metadata[table_column_metadata['table_name'] == right_table_name]

    for _, l_row in left_cols.iterrows():
        for _, r_row in right_cols.iterrows():
            if l_row['provided_data_type'] != r_row['provided_data_type']:
                continue

            criteria, confidence, threshold = None, None, 0

            if l_row['column_name'].lower() == r_row['column_name'].lower():
                criteria, threshold = 'Match on column name', FK_MATCH_THRESHOLDS['name']
            elif compute_similarity(l_row['column_description'], r_row['column_description']) > COL_DESC_MATCH_THRESHOLD:
                criteria, threshold = 'Match on column description', FK_MATCH_THRESHOLDS['desc']
            else:
                criteria, threshold = 'Match on column values', FK_MATCH_THRESHOLDS['value']

            match_percent = get_fk_match_percent(conn, left_table_name, l_row['column_name'], right_table_name, r_row['column_name'])
            
            if match_percent >= threshold:
                confidence = 'High' if match_percent >= 0.95 else ('Medium' if match_percent >= 0.8 else 'Low')
                relationships.append({
                    'left_table_name': left_table_name, 'left_column_name': l_row['column_name'],
                    'right_table_name': right_table_name, 'right_column_name': r_row['column_name'],
                    'join_criteria': criteria, 'match_confidence': confidence
                })
    return pd.DataFrame(relationships)

def create_knowledge_layer_table_relation(conn, input_data_dictionary_path, column_enriched_metadata):
    """Orchestrates the search for relationships between all table pairs."""
    table_information = standardize_column_name(pd.read_excel(input_data_dictionary_path, sheet_name='Table Information'))
    table_names = table_information['table_name'].values
    all_relationships = pd.DataFrame()

    if len(table_names) < 2: return all_relationships

    for left_table, right_table in permutations(table_names, 2):
        print(f'Finding relationships between: {left_table} -> {right_table}')
        relations = foreign_key_search(conn, left_table, right_table, column_enriched_metadata)
        all_relationships = pd.concat([all_relationships, relations], ignore_index=True)
        
    return all_relationships

def generate_column_info_for_table(conn, table_name, table_description, column_definitions_df, api_key, api_url):
    """Generates the full 'Column Information' DataFrame for a SINGLE table."""
    print(f"--- Generating Column Information for table: {table_name} ---")

    required_columns_df = column_definitions_df[column_definitions_df['table_name'] == table_name]
    required_columns_list = required_columns_df['column_name'].tolist()
    if not required_columns_list: return pd.DataFrame()

    full_schema_df = extract_schema(conn, table_name)
    if full_schema_df.empty: return pd.DataFrame()

    column_metadata_df = full_schema_df[full_schema_df['col_name'].isin(required_columns_list)].copy()
    column_descriptions = required_columns_df[['table_name', 'column_name', 'column_description']]
    
    column_value_summary = get_column_value_summary(conn, table_name, column_metadata_df)
    if column_value_summary.empty: return pd.DataFrame()
        
    sample_table_data = column_value_summary[['column_name', 'example_values']]
    column_role_datatype = infer_column_datatype_role(conn, table_name, column_metadata_df, column_descriptions, sample_table_data, api_key, api_url)
    
    merged = pd.merge(column_value_summary, column_role_datatype, on=['table_name', 'column_name'], how='left')
    merged['missing_value_percent'] = ((merged['missing_values'] / merged['total_values']) * 100).round(2).fillna(0)
    
    with_desc = pd.merge(merged, column_descriptions, on=['table_name', 'column_name'], how='left')
    
    enriched = get_column_metadata_from_llm(table_name, table_description, with_desc, api_key, api_url)
    final_info = derive_overall_column_confidence(enriched)
    
    # Ensure all columns exist
    for col in COLUMN_INFORMATION_OUTPUT_COLUMN_ORDER:
        if col not in final_info.columns: final_info[col] = None
            
    return final_info[COLUMN_INFORMATION_OUTPUT_COLUMN_ORDER].copy()

def generate_table_info_for_table(conn, table_name, table_description, single_table_column_info_df, api_key, api_url):
    """Generates the 'Table Information' DataFrame (one row) for a SINGLE table."""
    print(f"--- Generating Table Information for table: {table_name} ---")

    pk_candidates = get_primary_key(conn, table_name, table_description, single_table_column_info_df)
    other_table_info = enrich_table_information_with_llm(table_name, table_description, single_table_column_info_df, api_key, api_url)
    
    sample_data = execute_query(conn, f'SELECT * FROM "{table_name}" LIMIT 100')
    table_summary = get_table_summary_from_llm(table_name, table_description, single_table_column_info_df, sample_data, api_key, api_url)
    
    table_info = {
        "table_name": table_name, "table_description": table_description, "primary_key": pk_candidates,
        "table_name_synonyms": other_table_info.get("table_name_synonyms", []),
        "fact_or_dimension": other_table_info.get("fact_or_dimension", "NA"),
    }
    table_info.update(table_summary)
    
    return pd.DataFrame([table_info])

def append_df_to_excel_sheet(excel_path, df_to_append, sheet_name):
    """Appends a DataFrame to a specific sheet in an Excel file."""
    try:
        with pd.ExcelFile(excel_path) as xls:
            all_sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    except FileNotFoundError:
        all_sheets = {}

    existing_df = all_sheets.get(sheet_name)
    
    if existing_df is not None:
        # Standardize columns for safe concatenation
        df_to_append.columns = df_to_append.columns.str.lower().str.replace(' ', '_')
        existing_df.columns = existing_df.columns.str.lower().str.replace(' ', '_')
        combined_df = pd.concat([existing_df, df_to_append], ignore_index=True)
    else:
        combined_df = df_to_append

    all_sheets[sheet_name] = combined_df

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for name, data in all_sheets.items():
            beautified_data = beautify_column_names(data.copy())
            beautified_data.to_excel(writer, sheet_name=name, index=False)
    
    print(f"Successfully appended {len(df_to_append)} rows to '{sheet_name}' in {excel_path}")


# --- Main Orchestration Function ---

def create_knowledge_layer(conn, filled_dict_path, output_path, api_key, api_url):
    """Main orchestration function for generating the final knowledge layer."""
    print("\n--- Starting: Knowledge Layer Generation Stage ---")
    
    try:
        dw_info = pd.read_excel(filled_dict_path, sheet_name='DW Information')
        table_info_input = standardize_column_name(pd.read_excel(filled_dict_path, sheet_name='Table Information'))
        column_info_input = standardize_column_name(pd.read_excel(filled_dict_path, sheet_name='Column Information'))
    except FileNotFoundError:
        print(f"FATAL ERROR: Filled dictionary not found at {filled_dict_path}.")
        return

    if os.path.exists(output_path):
        os.remove(output_path)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        beautify_column_names(dw_info.copy()).to_excel(writer, sheet_name="DW Information", index=False)
    
    all_column_info_collected_df = pd.DataFrame()

    for _, row in table_info_input.iterrows():
        table_name = row['table_name']
        table_description = row['table_description']
        
        single_table_column_info = generate_column_info_for_table(conn, table_name, table_description, column_info_input, api_key, api_url)
        
        if not single_table_column_info.empty:
            append_df_to_excel_sheet(output_path, single_table_column_info, "Column Information")
            all_column_info_collected_df = pd.concat([all_column_info_collected_df, single_table_column_info], ignore_index=True)
        else:
            print(f"Skipping table info generation for '{table_name}' due to missing column info.")
            continue

        single_table_info = generate_table_info_for_table(conn, table_name, table_description, single_table_column_info, api_key, api_url)
        
        if not single_table_info.empty:
            append_df_to_excel_sheet(output_path, single_table_info, "Table Information")

    # Final Step: Generate relationships
    print("\n--- All tables processed. Generating final Table Relationship sheet. ---")
    if not all_column_info_collected_df.empty:
        table_relation_df = create_knowledge_layer_table_relation(conn, filled_dict_path, all_column_info_collected_df)
        if not table_relation_df.empty:
            append_df_to_excel_sheet(output_path, table_relation_df, "Table Relationship")
        else:
            print("No table relationships were found.")
    
    print("\n--- Knowledge layer generation complete. ---")


# --- Entrypoint for Streamlit App ---

def run_generation(uploaded_files, warehouse_info, table_descriptions, api_key, api_url, folder_path):
    """
    The main entry point called by the Streamlit application.
    Orchestrates the entire process from user input to final output file.
    """
    # --- 0. Setup Environment ---
    start_time = time.time()
    temp_dir = os.path.join(folder_path, "data_knowledge")
    csv_dir = os.path.join(temp_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)

    initial_dict_path = os.path.join(temp_dir, "Data_Dictionary_Initial.xlsx")
    filled_dict_path = os.path.join(temp_dir, "Data_Dictionary_Filled.xlsx")
    final_output_path = os.path.join(temp_dir, f"Data_Knowledge_Layer_Output_{int(start_time)}.xlsx")
    db_path = os.path.join(temp_dir, "knowledge_layer.db")

    # --- 1. Prepare Initial Data Dictionary from User Input ---
    print("--- Preparing Initial Data Dictionary from UI ---")
    
    # Save uploaded files and get table/column names
    table_names = []
    all_columns_data = []
    for uploaded_file in uploaded_files:
        table_name = os.path.splitext(uploaded_file.name)[0]
        table_names.append(table_name)
        file_path = os.path.join(csv_dir, f"{table_name}.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)
        for col_name in df.columns:
            user_desc = table_descriptions.get(table_name, {}).get('columns', {}).get(col_name, '')
            all_columns_data.append({
                "table_name": table_name,
                "column_name": col_name,
                "column_description": user_desc
            })

    # Create DataFrames for the initial dictionary
    dw_info_df = pd.DataFrame([{"Data Warehouse Description": warehouse_info.get('description', '')}])
    table_info_list = [{"table_name": name, "table_description": table_descriptions.get(name, {}).get('description', '')} for name in table_names]
    table_info_df = pd.DataFrame(table_info_list)
    column_info_df = pd.DataFrame(all_columns_data)

    # Write the initial Excel file
    with pd.ExcelWriter(initial_dict_path, engine='openpyxl') as writer:
        beautify_column_names(dw_info_df.copy()).to_excel(writer, sheet_name='DW Information', index=False)
        beautify_column_names(table_info_df.copy()).to_excel(writer, sheet_name='Table Information', index=False)
        beautify_column_names(column_info_df.copy()).to_excel(writer, sheet_name='Column Information', index=False)
    
    print(f"Initial dictionary saved to {initial_dict_path}")

    # --- 2. Setup Database ---
    conn = setup_database_from_csvs(db_path, csv_dir, table_names)
    if not conn:
        raise Exception("Database setup failed. Cannot proceed.")

    try:
        # --- 3. Fill Missing Descriptions ---
        success = fill_missing_descriptions(conn, initial_dict_path, filled_dict_path, api_key, api_url,warehouse_info)
        if not success:
            raise Exception("Failed to fill missing descriptions.")

        # --- 4. Generate the Full Knowledge Layer ---
        create_knowledge_layer(conn, filled_dict_path, final_output_path, api_key, api_url)

    except Exception as e:
        print(f"An error occurred during the main process: {e}")
        raise e
    finally:
        # --- 5. Cleanup ---
        conn.close()
        shutil.rmtree(csv_dir)
        print("Database connection closed.")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    return final_output_path , processing_time