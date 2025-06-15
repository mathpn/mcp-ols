import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from statsmodels.api import formula as smf


class DataAnalysisServer:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "data": None,
            "metadata": {},
            "created_at": datetime.now(),
        }
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
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
def load_data(session_id: str, file_path: Path) -> str:
    """Load data into a specific session"""
    assert isinstance(file_path, Path)
    session = server.get_session(session_id)
    session["data"] = pd.read_csv(file_path)
    session["metadata"]["file_path"] = file_path
    return f"Data loaded successfully into session {session_id}"


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
