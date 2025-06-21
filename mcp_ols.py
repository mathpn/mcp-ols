import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mcp.server.fastmcp import FastMCP, Image
from scipy import stats
from sqlalchemy import create_engine
from statsmodels.api import formula as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

# Set matplotlib to non-interactive backend
plt.switch_backend("Agg")
sns.set_style("whitegrid")


class DataAnalysisServer:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "data": None,
            "metadata": {},
            "models": {},
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
def run_ols_regression(session_id: str, formula: str):
    """Run a linear regression based on a patsy formula.

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    session = server.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    model = smf.ols(formula, data).fit()

    # Store the fitted model in the session
    model_id = f"ols_{len(session['models']) + 1}"
    session["models"][model_id] = {"model": model, "formula": formula, "type": "ols"}

    return {"model_id": model_id, "summary": model.summary().as_html()}


@mcp.tool()
def run_logistic_regression(session_id: str, formula: str):
    """Run a logistic regression based on a patsy formula.

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    session = server.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    model = smf.logit(formula, data).fit()

    # Store the fitted model in the session
    model_id = f"logit_{len(session['models']) + 1}"
    session["models"][model_id] = {"model": model, "formula": formula, "type": "logit"}

    return {"model_id": model_id, "summary": model.summary().as_html()}


@mcp.tool()
def describe_data(session_id: str) -> str:
    """Describe data loaded in the data frame."""
    session = server.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    return data.dtypes


def _get_residuals(model_info):
    match model_info["type"]:
        case "ols":
            return model_info["model"].resid
        case "logit":
            return model_info["model"].resid_response
        case _:
            raise NotImplementedError("unsupported model type")


@mcp.tool()
def create_residual_plots(session_id: str, model_id: str) -> Image:
    """Create residual diagnostic plots for a fitted model.

    Returns base64-encoded images of residual plots.
    """
    session = server.get_session(session_id)
    if model_id not in session["models"]:
        raise ValueError(f"Model {model_id} not found in session")

    model_info = session["models"][model_id]
    model = model_info["model"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Residual Diagnostics for {model_id}", fontsize=16)

    # 1. Residuals vs Fitted
    resid = _get_residuals(model_info)
    axes[0, 0].scatter(model.fittedvalues, resid, alpha=0.6)
    axes[0, 0].axhline(y=0, color="red", linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted")

    # 2. Q-Q Plot
    stats.probplot(_get_residuals(model_info), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot")

    # 3. Scale-Location Plot
    sqrt_abs_resid = np.sqrt(np.abs(model.resid_pearson))
    axes[1, 0].scatter(model.fittedvalues, sqrt_abs_resid, alpha=0.6)
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("√|Standardized Residuals|")
    axes[1, 0].set_title("Scale-Location")

    # 4. Residuals vs Leverage
    leverage = model.get_influence().hat_matrix_diag
    axes[1, 1].scatter(leverage, model.resid_pearson, alpha=0.6)
    axes[1, 1].axhline(y=0, color="red", linestyle="--")
    axes[1, 1].set_xlabel("Leverage")
    axes[1, 1].set_ylabel("Standardized Residuals")
    axes[1, 1].set_title("Residuals vs Leverage")

    plt.tight_layout()
    bytes_io = io.BytesIO()
    plt.savefig(bytes_io, format="png")
    return Image(data=bytes_io.getvalue(), format="png")


@mcp.tool()
def model_assumptions_test(session_id: str, model_id: str) -> str:
    """Test model assumptions for a fitted regression model."""
    session = server.get_session(session_id)
    if model_id not in session["models"]:
        raise ValueError(f"Model {model_id} not found in session")

    model_info = session["models"][model_id]
    model = model_info["model"]

    results = []
    results.append(f"Model Assumption Tests for {model_id}")

    # Normality test (Jarque-Bera)
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(_get_residuals(model_info))
    results.append("\nNormality Test (Jarque-Bera):")
    results.append(f"  Statistic: {jb_stat:.4f}")
    results.append(f"  P-value: {jb_pvalue:.4f}")
    results.append(
        f"  Result: {'Normal' if jb_pvalue > 0.05 else 'Not Normal'} (α=0.05)"
    )

    # Homoscedasticity test (Breusch-Pagan)
    if model_info["type"] == "ols":
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(
            _get_residuals(model_info), model.model.exog
        )
        results.append("\nHomoscedasticity Test (Breusch-Pagan):")
        results.append(f"  Statistic: {bp_stat:.4f}")
        results.append(f"  P-value: {bp_pvalue:.4f}")
        results.append(
            f"  Result: {'Homoscedastic' if bp_pvalue > 0.05 else 'Heteroscedastic'} (α=0.05)"
        )

    # Additional model statistics
    results.append("\nModel Statistics:")
    if hasattr(model, "rsquared"):
        results.append(f"  R-squared: {getattr(model, 'rsquared', 'N/A')}")
        results.append(f"  Adj. R-squared: {getattr(model, 'rsquared_adj', 'N/A')}")
    results.append(f"  AIC: {model.aic:.4f}")
    results.append(f"  BIC: {model.bic:.4f}")

    return "\n".join(results)


@mcp.tool()
def vif_table(session_id: str, model_id: str):
    """
    Compute a variance inflation factor (VIF) table.

    VIF is a measure of multicollinearity.
    VIF > 5 for a variable indicates that it is highly collinear with the
    other input variables.
    """
    session = server.get_session(session_id)
    if model_id not in session["models"]:
        raise ValueError(f"Model {model_id} not found in session")

    model_info = session["models"][model_id]
    model = model_info["model"]

    xvar = model.model.exog
    xvar_names = model.model.exog_names
    vif_df = pd.DataFrame()
    vif_df["Features"] = xvar_names
    vif_df["VIF Factor"] = [
        variance_inflation_factor(xvar, i) for i in range(xvar.shape[1])
    ]

    return vif_df.sort_values("VIF Factor").round(2).to_markdown(index=False)


@mcp.tool()
def influence_diagnostics(session_id: str, model_id: str):
    """Create influence diagnostics plot for a fitted model."""
    session = server.get_session(session_id)
    if model_id not in session["models"]:
        raise ValueError(f"Model {model_id} not found in session")

    model_info = session["models"][model_id]
    model = model_info["model"]

    # Get influence measures
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Influence Diagnostics for {model_id}", fontsize=16)

    # Cook's Distance
    ax1.stem(range(len(cooks_d)), cooks_d, basefmt=" ")
    ax1.axhline(
        y=4 / len(cooks_d), color="red", linestyle="--", label="Threshold (4/n)"
    )
    ax1.set_xlabel("Observation")
    ax1.set_ylabel("Cook's Distance")
    ax1.set_title("Cook's Distance")
    ax1.legend()

    # Leverage vs Standardized Residuals
    ax2.scatter(leverage, model.resid_pearson, alpha=0.6)
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.axvline(
        x=2 * len(model.params) / len(leverage), color="red", linestyle="--", alpha=0.5
    )
    ax2.set_xlabel("Leverage")
    ax2.set_ylabel("Standardized Residuals")
    ax2.set_title("Leverage vs Standardized Residuals")

    plt.tight_layout()
    bytes_io = io.BytesIO()
    plt.savefig(bytes_io, format="png")
    return Image(data=bytes_io.getvalue(), format="png")


@mcp.tool()
def create_partial_dependence_plot(
    session_id: str, model_id: str, feature: str, num_points: int = 100
) -> Image:
    """Create a partial dependence plot (PDP) for a specific feature.

    A partial dependence plot shows the marginal effect of a feature on the predicted
    outcome of a model. It shows how the model's predictions change as a feature varies
    over its range, while keeping all other features constant.

    Args:
        session_id: The ID of the analysis session
        model_id: The ID of the fitted model to analyze
        feature: The name of the feature to analyze
        num_points: Number of points to evaluate the partial dependence (default: 100)

    Returns:
        A matplotlib figure showing the partial dependence plot

    Raises:
        ValueError: If the session, model, or feature is not found
    """
    session = server.get_session(session_id)
    if model_id not in session["models"]:
        raise ValueError(f"Model {model_id} not found in session")

    model_info = session["models"][model_id]
    model = model_info["model"]
    data = session["data"]

    if feature not in data.columns:
        raise ValueError(f"Feature {feature} not found in dataset")

    # Create a range of values for the feature
    feature_values = np.linspace(data[feature].min(), data[feature].max(), num_points)

    # Calculate partial dependence
    avg_predictions = []
    for value in feature_values:
        # Create a copy of the data with the feature set to the current value
        temp_data = data.copy()
        temp_data[feature] = value

        # Get predictions
        if model_info["type"] == "ols":
            predictions = model.predict(temp_data)
        else:  # logistic regression
            predictions = model.predict(temp_data)

        # Average the predictions
        avg_predictions.append(predictions.mean())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(feature_values, avg_predictions)
    ax.set_xlabel(feature)
    ax.set_ylabel("Partial dependence")
    ax.set_title(f"Partial Dependence Plot for {feature}")

    # Add rug plot
    ax.plot(
        data[feature], np.zeros_like(data[feature]) - 0.1, "|", color="k", alpha=0.2
    )

    # Add confidence intervals if it's an OLS model
    if model_info["type"] == "ols":
        std_dev = np.std(avg_predictions)
        ax.fill_between(
            feature_values,
            np.array(avg_predictions) - 1.96 * std_dev,
            np.array(avg_predictions) + 1.96 * std_dev,
            alpha=0.2,
            label="95% CI",
        )
        ax.legend()

    plt.tight_layout()
    bytes_io = io.BytesIO()
    plt.savefig(bytes_io, format="png")
    plt.close(fig)
    return Image(data=bytes_io.getvalue(), format="png")
def list_models(session_id: str) -> list[dict[str, Any]]:
    """List all fitted models in a session.

    Returns:
        A list of dictionaries containing model information
    """
    session = server.get_session(session_id)

    results = []

    for model_id, model_info in session["models"].items():
        model = model_info["model"]
        results.append(
            {
                "model_id": model_id,
                "type": model_info["type"],
                "formula": model_info["formula"],
                "aic": model.aic,
            }
        )
    return results


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
