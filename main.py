from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from statsmodels.api import formula as smf

mcp = FastMCP("linear-regression")

cache = []


@mcp.tool()
def run_ols_regression(formula: str) -> str:
    """Run a linear regression based on a patsy formula

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    model = smf.ols(formula, cache[0]).fit()
    return model.summary()


@mcp.tool()
def run_logistic_regression(formula: str) -> str:
    """Run a logistic regression based on a patsy formula

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    model = smf.logit(formula, cache[0]).fit()
    return model.summary()


@mcp.tool()
async def load_data(file_path: Path) -> str:
    """Load data from a file path to a data frame.

    Args:
        file_path: path to the data file
    """
    assert isinstance(file_path, Path)
    df = pd.read_csv(file_path)
    cache.append(df)
    return "Loaded data succesfully"


@mcp.tool()
async def describe_data() -> str:
    """Describe data loaded in the data frame."""
    return cache[0].dtypes


if __name__ == "__main__":
    mcp.run(transport="stdio")
