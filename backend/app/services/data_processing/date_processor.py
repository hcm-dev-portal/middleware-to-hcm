# backend/app/services/data_processing/date_processor.py
from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DateProcessor:
    """Handles date-related processing and relative date conversions."""
    
    def __init__(self, data_anchor: Optional[str] = None):
        """
        Initialize DateProcessor.
        
        Args:
            data_anchor: Latest available data date in format 'YYYY-MM-DD'
        """
        self.data_anchor = data_anchor  # Format: 'YYYY-MM-DD'
    
    def set_data_anchor(self, anchor: str):
        """
        Set the data anchor date (latest available data date).
        
        Args:
            anchor: Date string in format 'YYYY-MM-DD'
        """
        self.data_anchor = anchor
        logger.info("Data anchor set to: %s", anchor)
    
    def rewrite_relative_dates(self, english_query: str) -> str:
        """
        Convert relative date expressions to absolute dates based on data anchor.
        
        This method handles common relative date expressions like:
        - "today" -> actual latest data date
        - "this month" -> year-month of latest data
        - "current day/month" -> corresponding absolute dates
        
        Args:
            english_query: Query text potentially containing relative dates
            
        Returns:
            Query text with relative dates converted to absolute dates
        """
        if not self.data_anchor:
            logger.debug("No data anchor set, returning query unchanged")
            return english_query
        
        anchor = self.data_anchor  # e.g., '2023-03-31'
        result = english_query
        
        # Replace 'today' and 'current day' with anchor date
        result = re.sub(
            r"\btoday\b|\bcurrent day\b", 
            anchor, 
            result, 
            flags=re.IGNORECASE
        )
        
        # Replace 'this month' and 'current month' with anchor's month
        if re.search(r"\bthis month\b|\bcurrent month\b", result, flags=re.IGNORECASE):
            try:
                yyyy, mm, _ = anchor.split("-")
                result = re.sub(
                    r"\bthis month\b|\bcurrent month\b",
                    f"{yyyy}-{mm} (month)", 
                    result, 
                    flags=re.IGNORECASE
                )
            except ValueError:
                logger.warning("Invalid data anchor format: %s", anchor)
        
        # Log if any changes were made
        if result != english_query:
            logger.debug("Rewrote relative dates: '%s' -> '%s'", english_query, result)
        
        return result
    
    def get_data_anchor(self) -> Optional[str]:
        """Get the current data anchor date."""
        return self.data_anchor
    
    def has_data_anchor(self) -> bool:
        """Check if a data anchor is set."""
        return self.data_anchor is not None
    
    def rewrite_sql_dates(self, sql: str) -> str:
        """
        Replace SQL date functions with anchor date for consistent results.
        
        This replaces GETDATE() calls with the actual anchor date to ensure
        queries return consistent results based on available data.
        
        Args:
            sql: SQL query string
            
        Returns:
            SQL with GETDATE() replaced by anchor date
        """
        if not self.data_anchor:
            return sql
        
        # Replace CAST(GETDATE() AS date) with anchor date
        result = re.sub(
            r"CAST\s*\(\s*GETDATE\(\)\s*AS\s*date\s*\)",
            f"'{self.data_anchor}'", 
            sql, 
            flags=re.IGNORECASE
        )
        
        # Replace plain GETDATE() calls
        result = re.sub(
            r"\bGETDATE\(\)",
            f"'{self.data_anchor}'", 
            result, 
            flags=re.IGNORECASE
        )
        
        return result
    
    def extract_date_range_from_query(self, query: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract explicit date ranges from natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (start_date, end_date) if found, otherwise (None, None)
        """
        # Look for date patterns like YYYY-MM-DD
        date_pattern = r'\b(\d{4})-(\d{2})-(\d{2})\b'
        dates = re.findall(date_pattern, query)
        
        if len(dates) >= 2:
            # If multiple dates found, assume first is start, last is end
            start_date = f"{dates[0][0]}-{dates[0][1]}-{dates[0][2]}"
            end_date = f"{dates[-1][0]}-{dates[-1][1]}-{dates[-1][2]}"
            return start_date, end_date
        elif len(dates) == 1:
            # Single date found, use as both start and end
            single_date = f"{dates[0][0]}-{dates[0][1]}-{dates[0][2]}"
            return single_date, single_date
        
        return None, None
    
    def is_relative_date_query(self, query: str) -> bool:
        """
        Check if query contains relative date expressions.
        
        Args:
            query: Query text to check
            
        Returns:
            True if query contains relative date expressions
        """
        relative_patterns = [
            r"\btoday\b",
            r"\byesterday\b", 
            r"\btomorrow\b",
            r"\bthis (week|month|year)\b",
            r"\blast (week|month|year)\b",
            r"\bnext (week|month|year)\b",
            r"\bcurrent (day|week|month|year)\b",
            r"\b\d+\s*(day|days|week|weeks|month|months)\s+(ago|from now)\b"
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in relative_patterns)