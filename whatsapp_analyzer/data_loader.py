# data_loader.py

import re
import pandas as pd
from tqdm import tqdm
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────
# Pattern A: 12-hour clock  →  01/03/20, 3:21 pm - Name: text   (English export)
# Pattern B: 24-hour clock  →  21/03/20, 11:55 - Name: text      (Italian export)
# Both accept 2- or 4-digit year; the date separator can be / or .

_DATE = r'(\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4})'
_TIME_12 = r'(\d{1,2}:\d{2}\s*[apAP][mM])'   # e.g. 3:21 pm
_TIME_24 = r'(\d{1,2}:\d{2})'                  # e.g. 11:55
_SEP = r'\s*[-–]\s*'

# Named-group pattern that tries 12-h first, falls back to 24-h
PATTERN = PATTERN = re.compile(
    r'^(\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4}),\s+(\d{1,2}:\d{2})\s*-\s*(?:(.*?):\s)?(.+)'
)


def _parse_line(line: str):
    m = PATTERN.match(line)
    if not m:
        return None

    g = m.groups()

    for base in (0, 4, 8, 12):
        if base + 3 < len(g):
            date = g[base]
            time = g[base + 1]
            name = g[base + 2]
            text = g[base + 3]

            # Skip if essential fields missing
            if not date or not time or not text:
                continue

            return (
                date.strip(),
                time.strip(),
                (name.strip() if name else "System"),
                text.strip()
            )

    return None


def _to_datetime(date_str: str, time_str: str) -> pd.Timestamp:
    """Try multiple date/time format combinations and return a Timestamp (NaT on failure)."""
    combined = f"{date_str} {time_str}"
    formats = [
        '%d/%m/%y %I:%M %p',   # 21/03/20 11:55 pm  – Italian 12-h (unlikely but safe)
        '%d/%m/%Y %I:%M %p',   # 21/03/2020 11:55 pm
        '%m/%d/%y %I:%M %p',   # 03/21/20 11:55 pm  – US 12-h
        '%m/%d/%Y %I:%M %p',   # 03/21/2020 11:55 pm
        '%d/%m/%y %H:%M',      # 21/03/20 11:55     – Italian 24-h ✓
        '%d/%m/%Y %H:%M',      # 21/03/2020 11:55
        '%m/%d/%y %H:%M',      # 03/21/20 11:55
        '%m/%d/%Y %H:%M',      # 03/21/2020 11:55
        '%d.%m.%y %H:%M',      # 21.03.20 11:55     – dot separator
        '%d.%m.%Y %H:%M',
        '%d.%m.%y %I:%M %p',
        '%d.%m.%Y %I:%M %p',
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(combined, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def load_and_parse_chat(file_path_or_uploaded_file, use_gpu=False) -> Optional[pd.DataFrame]:
    """
    Load and parse a WhatsApp chat export (English or Italian format).

    Supports:
    - 12-hour clock (English):  01/03/20, 3:21 pm - Name: text
    - 24-hour clock (Italian):  21/03/20, 11:55 - Name: text
    - 2-digit and 4-digit years
    - '/' and '.' date separators
    """
    data = []
    current_entry = None

    try:
        if hasattr(file_path_or_uploaded_file, 'read'):
            logger.info("Processing uploaded file")
            raw = file_path_or_uploaded_file.getvalue()
            # Try UTF-8 first, then latin-1 (common for Italian exports)
            try:
                file_content = raw.decode('utf-8')
            except UnicodeDecodeError:
                file_content = raw.decode('latin-1')
            lines = file_content.splitlines()
        else:
            logger.info(f"Reading file from path: {file_path_or_uploaded_file}")
            try:
                with open(file_path_or_uploaded_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(file_path_or_uploaded_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

        for line in tqdm(lines, desc="Parsing chat"):
            line = line.strip()
            if not line:
                continue

            parsed = _parse_line(line)
            if parsed:
                if current_entry:
                    data.append(current_entry)
                date, time, name, text = parsed
                current_entry = {
                    'Date': date,
                    'Time': time,
                    'Name': name,
                    'Text': text,
                    'IsMedia': bool(re.search(r'<Media omitted>|<.+omesso>', text, re.IGNORECASE)),
                }
            elif current_entry:
                current_entry['Text'] += '\n' + line

        if current_entry:
            data.append(current_entry)

        if not data:
            logger.warning("No chat data was parsed from the file")
            return None

        df = pd.DataFrame(data)

        df['Datetime'] = df.apply(
            lambda row: _to_datetime(row['Date'], row['Time']), axis=1
        )

        invalid = df['Datetime'].isna().sum()
        if invalid > 0:
            logger.warning(f"Dropped {invalid} rows with unparseable dates")
            df = df.dropna(subset=['Datetime'])

        df['DayOfWeek'] = df['Datetime'].dt.day_name()
        df['Hour'] = df['Datetime'].dt.hour
        df['Date'] = df['Datetime'].dt.date

        logger.info(
            f"Successfully parsed {len(df)} messages from {df['Name'].nunique()} participants"
        )
        return df

    except Exception as e:
        logger.error(f"Error parsing chat data: {str(e)}")
        return None


def save_to_csv(df: pd.DataFrame, output_path: str) -> bool:
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to CSV: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse WhatsApp chat export to CSV")
    parser.add_argument("input_file")
    parser.add_argument("--output", "-o", default="whatsapp_chat.csv")
    args = parser.parse_args()
    df = load_and_parse_chat(args.input_file)
    if df is not None:
        save_to_csv(df, args.output)
    else:
        logger.error("Failed to parse chat data.")
