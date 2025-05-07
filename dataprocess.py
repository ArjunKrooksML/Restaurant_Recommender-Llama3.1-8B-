import pandas as pd
from langchain_core.documents import Document
import re

def loadcsv(csvfile: str):
    try:
        return pd.read_csv(csvfile)
    except FileNotFoundError:
        print(f"Error: File not found at {csvfile}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def reviews(revs: str):
    try:
        if pd.isna(revs) or not isinstance(revs, str):
            return ""
        review_texts = re.findall(r"\(\s*'(?:[^']*)'\s*,\s*'([^']*)'\s*\)", revs)
        return " ".join(review_texts)
    except Exception:
        return str(revs) if isinstance(revs, str) else ""

def doc_text(row: pd.Series):
    name = row.get('name', 'N/A')
    location = row.get('location', 'N/A')
    city = row.get('listed_in(city)', 'N/A')
    cuisines = row.get('cuisines', 'N/A')
    dish_liked = row.get('dish_liked', 'N/A')
    rest_type = row.get('rest_type', 'N/A')
    cost = row.get('approx_cost(for two people)', 'N/A')
    online = row.get('online_order', 'N/A')
    booking = row.get('book_table', 'N/A')
    rate = row.get('rate', 'N/A')
    votes = row.get('votes', 'N/A')
    user_rev = reviews(row.get('reviews_list', ''))

    return (
        f"Restaurant Name: {name}. Location: {location}, {city}. "
        f"Cuisines: {cuisines}. Dishes Liked: {dish_liked}. Type: {rest_type}. "
        f"Approx. Cost for Two: {cost}. Online Order: {online}. Table Booking: {booking}. "
        f"Rating: {rate}. Votes: {votes}. Reviews: {user_rev}"
    )

def process_data(df: pd.DataFrame):
    if df is None or df.empty:
        print("Warning: Input DataFrame is empty or None. Returning empty list.")
        return []

    unique_columns = ['name', 'address']
    columns_to_check = [col for col in unique_columns if col in df.columns]

    if columns_to_check:
        original_rows = len(df)
        df_unique = df.drop_duplicates(subset=columns_to_check, keep='first')
        dropped_rows = original_rows - len(df_unique)
        if dropped_rows > 0:
            print(f"Removed {dropped_rows} duplicate rows based on columns: {columns_to_check}")
    else:
        print(f"Warning: None of the specified unique columns {unique_columns} found in DataFrame. Skipping deduplication.")
        df_unique = df

    if df_unique.empty:
        print("Warning: DataFrame is empty after deduplication. Returning empty list.")
        return []

    docs = []
    for _, row in df_unique.iterrows():
        page_c = doc_text(row)
        meta = {
            "name": str(row.get('name', 'N/A')),
            "location": str(row.get('location', 'N/A')),
            "city": str(row.get('listed_in(city)', 'N/A')),
            "cuisines": str(row.get('cuisines', 'N/A')),
            "rate": str(row.get('rate', 'N/A')),
            "cost_for_two": str(row.get('approx_cost(for two people)', 'N/A')),
            "url": str(row.get('url', 'N/A'))
        }
        docs.append(Document(page_content=page_c, metadata=meta))

    return docs
