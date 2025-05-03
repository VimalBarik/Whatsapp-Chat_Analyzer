# data_loader.py

import re
import pandas as pd
from tqdm import tqdm
from io import StringIO
import logging
from typing import Union, Optional, IO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_parse_chat(file_path_or_uploaded_file, use_gpu=False) -> Optional[pd.DataFrame]:
    """
    Load and parse WhatsApp chat exported data into a pandas DataFrame.
    
    Args:
        file_path_or_uploaded_file: Either a file path string or an uploaded file object
        use_gpu: Whether to use GPU acceleration (not implemented)
        
    Returns:
        DataFrame with parsed chat data or None if parsing fails
    """
    # WhatsApp chat message pattern (both user messages and system messages)
    pattern = re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}\s*[apAP][mM]\b)\s*-\s*'
        r'([^:]+):\s*(.+)|'
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}\s*[apAP][mM]\b)\s*-\s*(.+)'
    )
    
    data = []
    current_entry = None
    
    try:
        # Read file content - handle both file paths and uploaded file objects
        if hasattr(file_path_or_uploaded_file, "read"):
            logger.info("Processing uploaded file")
            file_content = file_path_or_uploaded_file.getvalue().decode("utf-8")
            lines = file_content.splitlines()
        else:
            logger.info(f"Reading file from path: {file_path_or_uploaded_file}")
            with open(file_path_or_uploaded_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
        # Process each line in the chat
        for line_num, line in enumerate(tqdm(lines, desc="Parsing chat")):
            line = line.strip()
            if not line:
                continue
                
            match = pattern.match(line)
            if match:
                # If we have a current entry, add it to our data before starting a new one
                if current_entry:
                    data.append(current_entry)
                    
                if match.group(1):  # User message
                    date, time, name, text = match.group(1, 2, 3, 4)
                else:  # System message
                    date, time, text = match.group(5, 6, 7)
                    name = "System"
                    
                current_entry = {
                    'Date': date,
                    'Time': time,
                    'Name': name.strip(),  # Clean any whitespace
                    'Text': text,
                    'IsMedia': bool(re.search(r"<Media omitted>", text, re.IGNORECASE))
                }
            elif current_entry:
                # This is a continuation of the previous message (multi-line)
                current_entry['Text'] += '\n' + line
                
        # Don't forget the last entry
        if current_entry:
            data.append(current_entry)
            
        # If we didn't parse any data, return None
        if not data:
            logger.warning("No chat data was parsed from the file")
            return None
            
        # Create DataFrame from parsed data
        df = pd.DataFrame(data)
        
        # Convert to datetime format
        try:
            df['Datetime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format='%d/%m/%Y %I:%M %p',  # Ensures correct parsing (e.g., "3:21 pm")
                errors='coerce'
            )
        except Exception as e:
            logger.warning(f"Date parsing failed with format '%d/%m/%Y %I:%M %p': {e}")
            # Try alternative format
            try:
                df['Datetime'] = pd.to_datetime(
                    df['Date'] + ' ' + df['Time'], 
                    format='%m/%d/%Y %I:%M %p',
                    errors='coerce'
                )
            except Exception as e:
                logger.warning(f"Alternative date parsing also failed: {e}")
        
        # Drop rows with invalid datetime
        invalid_dates = df['Datetime'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropped {invalid_dates} rows with invalid date format")
            df = df.dropna(subset=['Datetime'])
            
        # Add derived datetime columns
        df['DayOfWeek'] = df['Datetime'].dt.day_name()
        df['Hour'] = df['Datetime'].dt.hour
        df['Date'] = df['Datetime'].dt.date
        
        logger.info(f"Successfully parsed {len(df)} messages from {df['Name'].nunique()} participants")
        return df
        
    except Exception as e:
        logger.error(f"Error parsing chat data: {str(e)}")
        return None

def save_to_csv(df: pd.DataFrame, output_path: str) -> bool:
    """Save the parsed DataFrame to CSV"""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to CSV: {str(e)}")
        return False

# For command-line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse WhatsApp chat export to CSV")
    parser.add_argument("input_file", help="Path to WhatsApp chat export file")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    
    args = parser.parse_args()
    
    output_path = args.output or "whatsapp_chat.csv"
    df = load_and_parse_chat(args.input_file)
    
    if df is not None:
        save_to_csv(df, output_path)
    else:
        logger.error("Failed to parse chat data. No output generated.")