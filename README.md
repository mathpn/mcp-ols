# MCP-OLS: Statistical Analysis Server

A Model Context Protocol (MCP) server that provides statistical modeling and regression analysis capabilities to LLMs.
Built with FastMCP, this server enables data loading, statistical modeling, diagnostics, and visualization through a simple, session-based interface.

## Features

### Data Management

- **Multi-format data loading**: CSV, Excel (.xlsx/.xls), JSON, Parquet, and SQLite databases
- **Session-based workflow**: Create isolated analysis sessions for different datasets
- **Data description**: Dataset overview with statistics and missing value analysis

### Statistical Modeling

- **Linear Regression (OLS)**: Ordinary least squares regression with patsy formula syntax (`Y ~ X1 + X2 + X3`)
- **Logistic Regression**: Binary classification modeling
- **Model comparison**: Side-by-side comparison of multiple models with performance metrics

### Model Diagnostics

- **Residual analysis**: 4-panel diagnostic plots (residuals vs fitted, Q-Q plot, scale-location, leverage)
- **Assumption testing**: Jarque-Bera normality test and Breusch-Pagan homoscedasticity test
- **Influence diagnostics**: Cook's distance and leverage plots to identify outliers
- **Multicollinearity detection**: Variance Inflation Factor (VIF) analysis
- **Partial dependence plots**: Visualize feature effects on model predictions

## Installation & Usage

### MCP Client Configuration

To use this MCP server with common MCP clients, you need to add it to their configuration.
Here we describe how to add `mcp-ols` to Claude Desktop and VS Code, you can find how to configure MCP servers in other clients in their documentation.
It uses `uvx` to run the server, which requires `uv` installed.

#### Claude Desktop

Add to your Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "mcp-ols": {
      "command": "uvx",
      "args": ["mcp-ols"]
    }
  }
}
```

For local development:

```json
{
  "mcpServers": {
    "mcp-ols": {
      "mcp-ols": {
        "command": "uvx",
        "args": ["--refresh", "--from", "/path/to/repo", "mcp-ols"]
      }
  }
}
```

#### VS Code

Add to your workspace settings (`.vscode/settings.json`):

```json
{
  "mcp": {
    "servers": {
      "mcp-ols": {
        "command": "uvx",
        "args": ["mcp-ols"]
      }
    }
  }
}
```

For local development:

```json
{
  "mcp": {
    "servers": {
      "mcp-ols": {
        "command": "uvx",
        "args": ["--refresh", "--from", "/path/to/repo", "mcp-ols"]
      }
    }
  }
}
```

### Running with uvx (Recommended)

The easiest way to run the MCP server separate from a client is also through `uvx`, which handles dependencies automatically:

```bash
uvx mcp-ols
```

This command will automatically download and run the server with all required dependencies in an isolated environment.

### Full Installation for Development

For development:

```bash
git clone https://github.com/mathpn/mcp-ols
cd mcp-ols

# Install with development dependencies
uv sync --group dev

# Run the server
uv run mcp_ols.py
```

### Basic Usage Workflow

1. **Create a session**
2. **Load your data** from various formats
3. **Run statistical models** using formula syntax
4. **Generate diagnostics** and visualizations
5. **Compare models** and analyze results

## API Reference

### Core Tools

#### `create_analysis_session()`

Creates a new isolated analysis session.

- **Returns**: `{"session_id": "1"}`

#### `load_data(session_id: str, file_path: str)`

Loads data into a session from various formats.

- **Supported formats**: CSV, Excel, JSON, Parquet, SQLite
- **SQLite syntax**: `sqlite:///path/to/db.sqlite/table_name`

#### `describe_data(session_id: str)`

Returns dataset overview including data types, missing values, and summary statistics.

### Statistical Modeling

#### `run_ols_regression(session_id: str, formula: str)`

Performs linear regression using patsy formula syntax.

- **Example**: `"Sales ~ TV + Radio + Newspaper"`
- **Returns**: Model ID and HTML summary

#### `run_logistic_regression(session_id: str, formula: str)`

Performs logistic regression for binary outcomes.

- **Example**: `"passed ~ hours_studied + practice_exams"`

### Model Diagnostics

#### `create_residual_plots(session_id: str, model_id: str)`

Generates 4-panel residual diagnostic plots.

#### `model_assumptions_test(session_id: str, model_id: str)`

Tests model assumptions (normality, homoscedasticity).

#### `influence_diagnostics(session_id: str, model_id: str)`

Creates Cook's distance and leverage plots.

#### `vif_table(session_id: str, model_id: str)`

Computes Variance Inflation Factors for multicollinearity detection.

### Advanced Analysis

#### `create_partial_dependence_plot(session_id: str, model_id: str, feature: str)`

Creates partial dependence plots showing feature effects.

#### `compare_models(session_id: str, model_ids: List[str])`

Compares multiple models with performance metrics table.

#### `visualize_model_comparison(session_id: str, model_ids: list[str])`

Creates comprehensive model comparison visualizations.

#### `list_models(session_id: str)`

Lists all fitted models in a session with key statistics.

## Example Client Usage

```python
import asyncio
from fastmcp import Client
from mcp_ols import mcp

async def example():
    async with Client(mcp) as client:
        # Create session
        result = await client.call_tool("create_analysis_session", {})
        session_id = result["session_id"]

        # Load data
        await client.call_tool("load_data", {
            "session_id": session_id,
            "file_path": "data.csv"
        })

        # Run regression
        result = await client.call_tool("run_ols_regression", {
            "session_id": session_id,
            "formula": "Sales ~ TV + Radio"
        })
        model_id = result["model_id"]

        # Generate diagnostics
        await client.call_tool("create_residual_plots", {
            "session_id": session_id,
            "model_id": model_id
        })
```

## Development

### Project Structure

- `mcp_ols.py`: Main server implementation with all MCP tools
- `test_mcp_server.py`: Test suite
- `pyproject.toml`: Project configuration and dependencies

### Running Tests

```bash
uv run pytest

# Run specific test
uv run pytest test_mcp_server.py::test_ols_regression -v
```

### Development Commands

```bash
# Install development dependencies
uv sync --group dev

# Add new dependency
uv add <package>
```

### Debugging with MCP Inspector

```bash
uv run fastmcp dev mcp_ols.py
```

This launches [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector), a web-based interactive developer tool for testing and debugging MCP servers.

## Supported Data Formats

The server supports loading data from multiple file formats:

- **CSV files** (.csv) - with automatic dialect detection
- **Excel files** (.xlsx, .xls)
- **JSON files** (.json)
- **Parquet files** (.parquet)
- **SQLite databases** - using `sqlite:///path/to/db.sqlite/table_name` syntax

## Contributing

1. Fork the repository
1. Create a feature branch
1. Make your changes
1. Add tests for new functionality
1. Run the test suite
1. Commit your changes
1. Open a Pull Request

## License

This project is under the Apache-2.0 license.
