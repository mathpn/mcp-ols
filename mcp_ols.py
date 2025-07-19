import io
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from matplotlib.gridspec import GridSpec
from scipy import stats
from sqlalchemy import create_engine
from statsmodels.api import formula as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

# Set matplotlib to non-interactive backend
plt.switch_backend("Agg")
sns.set_style("whitegrid")


class DataAnalysisSession:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}
        self._next_id = 1

    def create_session(self) -> str:
        session_id = str(self._next_id)
        self._next_id += 1
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


mcp = FastMCP("mcp-ols")
_session = DataAnalysisSession()


@mcp.tool(exclude_args=["server_session"])
def create_analysis_session(server_session=None):
    """Create a new analysis session"""
    if server_session is None:
        server_session = _session
    session_id = server_session.create_session()
    return {"session_id": session_id}


@mcp.tool(exclude_args=["server_session"])
def load_data(
    session_id: str,
    file_path: str | Path,
    server_session=None,
) -> str:
    """Load data into a specific session from various file formats.

    Supported formats:
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - JSON files (.json)
    - Parquet files (.parquet)
    - SQLite databases (sqlite:/// prefix)
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)

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
            session["data"] = pd.read_csv(file_path, engine="python", sep=None)
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


@mcp.tool(exclude_args=["server_session"])
def run_ols_regression(session_id: str, formula: str, server_session=None):
    """Run a linear regression based on a patsy formula.

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    model = smf.ols(formula, data).fit()

    model_id = f"ols_{len(session['models']) + 1}"
    session["models"][model_id] = {"model": model, "formula": formula, "type": "ols"}

    return {"model_id": model_id, "summary": model.summary().as_html()}


@mcp.tool(exclude_args=["server_session"])
def run_logistic_regression(session_id: str, formula: str, server_session=None):
    """Run a logistic regression based on a patsy formula.

    Args:
        formula: string of format Y ~ X_1 + X_2 + ... + X_n
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    model = smf.logit(formula, data).fit()

    model_id = f"logit_{len(session['models']) + 1}"
    session["models"][model_id] = {"model": model, "formula": formula, "type": "logit"}

    return {"model_id": model_id, "summary": model.summary().as_html()}


@mcp.tool(exclude_args=["server_session"])
def describe_data(session_id: str, server_session=None) -> str:
    """Describe data loaded in the data frame.

    Returns a string containing:
    - Data types for each column
    - Basic statistics (count, mean, std, min, max)
    - Number of missing values
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
    if session["data"] is None:
        raise ValueError("No data loaded in this session")

    data = session["data"]
    sections = []

    sections.append("## Dataset Overview\n")
    sections.append(f"- Rows: {data.shape[0]:,}")
    sections.append(f"- Columns: {data.shape[1]:,}")

    sections.append("## Column Data Types\n")
    sections.append(data.dtypes.to_markdown())
    sections.append("\n")

    missing = data.isnull().sum()
    if missing.any():
        sections.append("## Missing Values\n")
        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": (missing.values / len(data) * 100).round(2),
            }
        )
        missing_df = missing_df[missing_df["Missing Count"] > 0]
        sections.append(missing_df.to_markdown(index=False))
        sections.append("\n")

    sections.append("## Summary Statistics\n")
    sections.append(data.describe().round(2).to_markdown())

    return "\n".join(sections)


def _get_residuals(model_info):
    match model_info["type"]:
        case "ols":
            return model_info["model"].resid
        case "logit":
            return model_info["model"].resid_response
        case _:
            raise NotImplementedError("unsupported model type")


@mcp.tool(exclude_args=["server_session"])
def create_residual_plots(session_id: str, model_id: str, server_session=None) -> Image:
    """Create residual diagnostic plots for a fitted model.

    Returns base64-encoded images of residual plots.
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
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


@mcp.tool(exclude_args=["server_session"])
def model_assumptions_test(session_id: str, model_id: str, server_session=None) -> str:
    """Test model assumptions for a fitted regression model."""
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
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


@mcp.tool(exclude_args=["server_session"])
def vif_table(session_id: str, model_id: str, server_session=None):
    """
    Compute a variance inflation factor (VIF) table.

    VIF is a measure of multicollinearity.
    VIF > 5 for a variable indicates that it is highly collinear with the
    other input variables.
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
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


@mcp.tool(exclude_args=["server_session"])
def influence_diagnostics(session_id: str, model_id: str, server_session=None):
    """Create influence diagnostics plot for a fitted model."""
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
    if model_id not in session["models"]:
        raise ValueError(f"Model {model_id} not found in session")

    model_info = session["models"][model_id]
    model = model_info["model"]

    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag

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


@mcp.tool(exclude_args=["server_session"])
def create_partial_dependence_plot(
    session_id: str,
    model_id: str,
    feature: str,
    num_points: int = 100,
    server_session=None,
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
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)
    if model_id not in session["models"]:
        raise ValueError(f"Model {model_id} not found in session")

    model_info = session["models"][model_id]
    model = model_info["model"]
    data = session["data"]

    if feature not in data.columns:
        raise ValueError(f"Feature {feature} not found in dataset")

    feature_values = np.linspace(data[feature].min(), data[feature].max(), num_points)

    avg_predictions = []
    ci_lower = []
    ci_upper = []

    for value in feature_values:
        temp_data = data.copy()
        temp_data[feature] = value

        predictions = model.get_prediction(temp_data)
        mean_prediction = predictions.predicted_mean
        ci = predictions.conf_int(alpha=0.05)

        avg_predictions.append(mean_prediction.mean())
        ci_lower.append(ci[:, 0].mean())
        ci_upper.append(ci[:, 1].mean())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(feature_values, avg_predictions)
    ax.set_xlabel(feature)
    ax.set_ylabel("Partial dependence")
    ax.set_title(f"Partial Dependence Plot for {feature}")

    # Add rug plot
    ax.plot(
        data[feature],
        np.zeros_like(data[feature]) + np.min(ci_lower) - 0.1,
        "|",
        color="k",
        alpha=0.2,
    )

    # Add confidence intervals if it's an OLS model
    if model_info["type"] == "ols":
        ax.fill_between(
            feature_values,
            ci_lower,
            ci_upper,
            alpha=0.2,
            label="95% CI",
        )
        ax.legend()

    plt.tight_layout()
    bytes_io = io.BytesIO()
    plt.savefig(bytes_io, format="png")
    plt.close(fig)
    return Image(data=bytes_io.getvalue(), format="png")


@mcp.tool(exclude_args=["server_session"])
def list_models(session_id: str, server_session=None) -> list[dict[str, Any]]:
    """List all fitted models in a session.

    Returns:
        A list of dictionaries containing model information
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)

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


@mcp.tool(exclude_args=["server_session"])
def compare_models(
    session_id: str,
    model_ids: list[str],
    server_session=None,
) -> str:
    """Compare multiple models using various metrics.

    Args:
        session_id: The ID of the analysis session
        model_ids: List of model IDs to compare

    Returns:
        A formatted string with model comparison results
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)

    for model_id in model_ids:
        if model_id not in session["models"]:
            raise ValueError(f"Model {model_id} not found in session")

    comparison_data = []

    for model_id in model_ids:
        model_info = session["models"][model_id]
        model = model_info["model"]

        metrics = {
            "Model ID": model_id,
            "Type": model_info["type"],
            "Formula": model_info["formula"],
            "AIC": model.aic,
            "BIC": model.bic,
            "Log-Likelihood": model.llf,
        }

        if model_info["type"] == "ols":
            metrics.update(
                {
                    "R-squared": model.rsquared,
                    "Adj. R-squared": model.rsquared_adj,
                    "MSE": ((model.resid**2).mean()),
                    "RMSE": np.sqrt((model.resid**2).mean()),
                }
            )
        elif model_info["type"] == "logit":
            metrics.update(
                {
                    "Pseudo R-squared": model.prsquared,
                    "Percent Correctly Predicted": model.pred_table()[0]
                    / model.pred_table().sum(),
                }
            )

        comparison_data.append(metrics)

    comparison_df = pd.DataFrame(comparison_data)

    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)

    result = comparison_df.to_markdown(index=False)
    if result is None:
        raise ValueError("Failed to generate markdown table")
    return result


@mcp.tool(exclude_args=["server_session"])
async def visualize_model_comparison(
    session_id: str,
    model_ids: list[str],
    server_session=None,
) -> Image:
    """Create visualization comparing multiple models.

    Creates plots comparing various aspects of the models including:
    - Performance metrics (AIC, BIC, R² etc.)
    - Residual distributions
    - Feature coefficients/importance

    Args:
        session_id: The ID of the analysis session
        model_ids: List of model IDs to compare

    Returns:
        A matplotlib figure comparing the models
    """
    if server_session is None:
        server_session = _session

    session = server_session.get_session(session_id)

    for model_id in model_ids:
        if model_id not in session["models"]:
            raise ValueError(f"Model {model_id} not found in session")

    fig = plt.figure(figsize=(12, 8))
    all_ols = all(m["type"] == "ols" for m in session["models"].values())
    n_cols = 3 if all_ols else 2
    gs = GridSpec(2, n_cols)

    # Performance Metrics Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    comparison_data = []
    metric_names = ["AIC", "BIC"]

    for model_id in model_ids:
        model_info = session["models"][model_id]
        model = model_info["model"]
        metrics = [model.aic, model.bic]
        comparison_data.append(metrics)

    comparison_array = np.array(comparison_data)
    x = np.arange(len(model_ids))
    width = 0.8 / len(metric_names)

    for i, metric in enumerate(metric_names):
        ax1.bar(x + i * width, comparison_array[:, i], width, label=metric)

    ax1.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax1.set_xticklabels(model_ids)
    ax1.set_title("Model Performance Metrics (AIC/BIC)")
    ax1.legend()

    if all_ols:
        ax1_r2 = fig.add_subplot(gs[0, 1])
        r2_data = []
        r2_metric_names = ["R²", "Adj. R²"]

        for model_id in model_ids:
            model_info = session["models"][model_id]
            model = model_info["model"]
            r2_data.append([model.rsquared, model.rsquared_adj])

        r2_array = np.array(r2_data)
        width = 0.8 / len(r2_metric_names)

        for i, metric in enumerate(r2_metric_names):
            ax1_r2.bar(x + i * width, r2_array[:, i], width, label=metric)

        ax1_r2.set_xticks(x + width * (len(r2_metric_names) - 1) / 2)
        ax1_r2.set_xticklabels(model_ids)
        ax1_r2.set_title("Model Performance Metrics (R²)")
        ax1_r2.set_ylim(0, 1)
        ax1_r2.legend()

    # Residual distributions
    ax2 = fig.add_subplot(gs[0, -1])
    for model_id in model_ids:
        model_info = session["models"][model_id]
        resid = _get_residuals(model_info)
        sns.kdeplot(data=resid, label=model_id, ax=ax2)

    ax2.set_title("Residual Distributions")
    ax2.set_xlabel("Residual Value")
    ax2.set_ylabel("Density")
    ax2.legend()

    # Feature coefficients / importance
    ax3 = fig.add_subplot(gs[1, :])

    all_features = set()
    for model_id in model_ids:
        model_info = session["models"][model_id]
        model = model_info["model"]
        features = model.model.exog_names[1:]  # Exclude intercept
        all_features.update(features)

    all_features = sorted(all_features)
    group_width = 0.8
    bar_width = group_width / len(model_ids)
    feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}

    for i, model_id in enumerate(model_ids):
        model_info = session["models"][model_id]
        model = model_info["model"]

        model_features = model.model.exog_names[1:]
        model_coefs = model.params[1:]

        x_positions = []
        heights = []

        for feature in all_features:
            if feature not in model_features:
                x_pos = (
                    feature_to_idx[feature]
                    + (i * bar_width)
                    - (group_width / 2)
                    + (bar_width / 2)
                )
                ax3.axvspan(
                    x_pos - bar_width / 2,
                    x_pos + bar_width / 2,
                    alpha=0.2,
                    color="gray",
                )
                continue

            try:
                idx = model_features.index(feature)
                coef = model_coefs[idx]
            except ValueError:
                coef = 0

            x_pos = (
                feature_to_idx[feature]
                + (i * bar_width)
                - (group_width / 2)
                + (bar_width / 2)
            )
            x_positions.append(x_pos)
            heights.append(coef)

        ax3.bar(x_positions, heights, bar_width, label=model_id, alpha=0.7)

    ax3.set_xticks(range(len(all_features)))
    ax3.set_xticklabels(all_features, rotation=45, ha="right")
    ax3.set_title("Feature Coefficients by Model")
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    bytes_io = io.BytesIO()
    plt.savefig(bytes_io, format="png")
    plt.close(fig)
    return Image(data=bytes_io.getvalue(), format="png")


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
