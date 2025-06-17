import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine
from statsmodels.api import formula as smf


class DataAnalysisServer:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "data": None,
            "metadata": {},
            "created_at": datetime.now(),
        }
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]


mcp = FastMCP("linear-regression")
server = DataAnalysisServer()


@mcp.tool()
def create_analysis_session() -> str:
    """Create a new analysis session"""
    return server.create_session()


@mcp.tool()
def load_data(session_id: str, file_path: str | Path) -> str:
    """Load data into a specific session from various file formats.

    Supported formats:
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - JSON files (.json)
    - Parquet files (.parquet)
    - SQLite databases (sqlite:/// prefix)
    """
    session = server.get_session(session_id)

    if isinstance(file_path, str):
        parsed = urlparse(file_path)
        if parsed.scheme == "sqlite":
            engine = create_engine(file_path)
            table = parsed.path.split("/")[-1]
            session["data"] = pd.read_sql_table(table, engine)
            session["metadata"]["file_path"] = file_path
            return f"Data loaded successfully from SQLite database into session {session_id}"
        file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            session["data"] = pd.read_csv(file_path)
        elif suffix in (".xlsx", ".xls"):
            session["data"] = pd.read_excel(file_path)
        elif suffix == ".json":
            session["data"] = pd.read_json(file_path)
        elif suffix == ".parquet":
            session["data"] = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        session["metadata"]["file_path"] = str(file_path)
        return f"Data loaded successfully into session {session_id}"

    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


@mcp.tool()
def run_ols_regression(session_id: str, formula: str) -> str:
    """Run a linear regression based on a patsy formula.

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    session = server.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    model = smf.ols(formula, data).fit()
    return model.summary()


@mcp.tool()
def run_logistic_regression(session_id: str, formula: str) -> str:
    """Run a logistic regression based on a patsy formula.

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    session = server.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    model = smf.logit(formula, data).fit()
    return model.summary()


@mcp.tool()
async def describe_data(session_id: str) -> str:
    """Describe data loaded in the data frame."""
    session = server.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    return data.dtypes


if __name__ == "__main__":
    mcp.run(transport="stdio")
