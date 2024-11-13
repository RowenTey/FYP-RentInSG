import io
import traceback
import logging
from utils.motherduckdb import db
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from cashews import cache

router = APIRouter()


@router.get("/")
@cache(ttl="6h")
async def retrieve_data(table: str = "property_listing"):
    try:
        # Query data as a DataFrame
        logging.info(f"Retrieving data from table {table}...")
        df = db.query_df(f"SELECT * FROM {table}")

        # Use an in-memory buffer to store CSV data
        logging.info("Converting DataFrame to CSV...")
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)  # Move cursor to the beginning of the buffer

        # Stream the CSV content as bytes
        response = StreamingResponse(
            content=iter([buffer.getvalue()]),  # Use an iterator to stream content
            status_code=200,
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={table}.csv"
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error retrieving data: {e}")
