import json
import os
import tempfile

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_ols import mcp


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return """TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9"""


@pytest.fixture
def sample_logistic_data():
    """Sample data suitable for logistic regression"""
    return """hours_studied,practice_exams,passed
2,1,0
4,2,1
6,3,1
1,0,0
8,4,1
3,1,0
7,3,1
5,2,1"""


@pytest.mark.asyncio
async def test_session_creation():
    """Test creating analysis sessions"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]
        assert isinstance(session_id, str)
        assert len(session_id) > 0

        result2 = await client.call_tool("create_analysis_session", {})
        session_id2 = json.loads(result2.content[0].text)["session_id"]
        assert session_id != session_id2


@pytest.mark.asyncio
async def test_data_loading(sample_csv_data):
    """Test loading data from CSV files"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            result = await client.call_tool(
                "load_data", {"session_id": session_id, "file_path": temp_path}
            )
            result = await client.call_tool("describe_data", {"session_id": session_id})
            description = result.content[0].text
            assert "TV" in description
            assert "Radio" in description
            assert "Sales" in description
        finally:
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_data_loading_errors():
    """Test error handling in data loading"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]

        # Test nonexistent file
        with pytest.raises(ToolError):
            await client.call_tool(
                "load_data",
                {"session_id": session_id, "file_path": "/nonexistent/file.csv"},
            )

        # Test invalid session
        with pytest.raises(ToolError):
            await client.call_tool(
                "load_data",
                {"session_id": "invalid-session", "file_path": "/tmp/file.csv"},
            )


@pytest.mark.asyncio
async def test_ols_regression(sample_csv_data):
    """Test OLS regression functionality"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            await client.call_tool(
                "load_data", {"session_id": session_id, "file_path": temp_path}
            )

            result = await client.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
            )
            model_info = json.loads(result.content[0].text)
            assert "model_id" in model_info
            assert "summary" in model_info
            assert "TV" in model_info["summary"]
            assert "Radio" in model_info["summary"]

            result2 = await client.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ TV"},
            )
            model_info2 = json.loads(result2.content[0].text)
            assert model_info["model_id"] != model_info2["model_id"]

            # Test invalid formula
            with pytest.raises(ToolError):
                await client.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "NonexistentColumn ~ TV"},
                )

        finally:
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_logistic_regression(sample_logistic_data):
    """Test logistic regression functionality"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_logistic_data)
            temp_path = f.name

        try:
            await client.call_tool(
                "load_data", {"session_id": session_id, "file_path": temp_path}
            )

            result = await client.call_tool(
                "run_logistic_regression",
                {
                    "session_id": session_id,
                    "formula": "passed ~ hours_studied + practice_exams",
                },
            )
            model_info = json.loads(result.content[0].text)
            assert "model_id" in model_info
            assert "summary" in model_info

        finally:
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_model_diagnostics(sample_csv_data):
    """Test model diagnostic functionality"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            await client.call_tool(
                "load_data", {"session_id": session_id, "file_path": temp_path}
            )

            result = await client.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
            )
            model_info = json.loads(result.content[0].text)
            model_id = model_info["model_id"]

            # Test residual plots
            result = await client.call_tool(
                "create_residual_plots",
                {"session_id": session_id, "model_id": model_id},
            )
            image_result = result.content[0]
            assert image_result.type == "image"
            assert len(image_result.data) > 0

            # Test model assumptions
            result = await client.call_tool(
                "model_assumptions_test",
                {"session_id": session_id, "model_id": model_id},
            )
            test_results = result.content[0].text
            assert "Jarque-Bera" in test_results
            assert "Breusch-Pagan" in test_results

            # Test influence diagnostics
            result = await client.call_tool(
                "influence_diagnostics",
                {"session_id": session_id, "model_id": model_id},
            )
            image_result = result.content[0]
            assert image_result.type == "image"
            assert len(image_result.data) > 0

            # Test VIF table
            result = await client.call_tool(
                "vif_table", {"session_id": session_id, "model_id": model_id}
            )
            vif_data = result.content[0].text
            assert "VIF" in vif_data

        finally:
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_model_comparison(sample_csv_data):
    """Test model comparison functionality"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            await client.call_tool(
                "load_data", {"session_id": session_id, "file_path": temp_path}
            )

            result1 = await client.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ TV"},
            )
            model1_id = json.loads(result1.content[0].text)["model_id"]

            result2 = await client.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
            )
            model2_id = json.loads(result2.content[0].text)["model_id"]

            # Test list models
            result = await client.call_tool("list_models", {"session_id": session_id})
            models_text = result.content[0].text
            assert model1_id in models_text
            assert model2_id in models_text

            # Test compare models
            result = await client.call_tool(
                "compare_models",
                {"session_id": session_id, "model_ids": [model1_id, model2_id]},
            )
            comparison = result.content[0].text
            assert "AIC" in comparison
            assert "BIC" in comparison

            # Test visualize model comparison
            result = await client.call_tool(
                "visualize_model_comparison",
                {"session_id": session_id, "model_ids": [model1_id, model2_id]},
            )
            image_result = result.content[0]
            assert image_result.type == "image"
            assert len(image_result.data) > 0

        finally:
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_partial_dependence_plots(sample_csv_data):
    """Test partial dependence plot functionality"""
    async with Client(mcp) as client:
        result = await client.call_tool("create_analysis_session", {})
        session_id = json.loads(result.content[0].text)["session_id"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_csv_data)
            temp_path = f.name

        try:
            await client.call_tool(
                "load_data", {"session_id": session_id, "file_path": temp_path}
            )

            result = await client.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
            )
            model_info = json.loads(result.content[0].text)
            model_id = model_info["model_id"]

            # Test partial dependence plot
            result = await client.call_tool(
                "create_partial_dependence_plot",
                {
                    "session_id": session_id,
                    "model_id": model_id,
                    "feature": "TV",
                },
            )
            image_result = result.content[0]
            assert image_result.type == "image"
            assert len(image_result.data) > 0

            # Test with nonexistent feature
            with pytest.raises(ToolError):
                await client.call_tool(
                    "create_partial_dependence_plot",
                    {
                        "session_id": session_id,
                        "model_id": model_id,
                        "feature": "NonexistentFeature",
                    },
                )

        finally:
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_tool_listing():
    """Test that all expected tools are available"""
    async with Client(mcp) as client:
        result = await client.list_tools()
        tool_names = [tool.name for tool in result]

        expected_tools = [
            "create_analysis_session",
            "load_data",
            "run_ols_regression",
            "run_logistic_regression",
            "describe_data",
            "create_residual_plots",
            "model_assumptions_test",
            "vif_table",
            "influence_diagnostics",
            "create_partial_dependence_plot",
            "visualize_model_comparison",
            "compare_models",
            "list_models",
        ]

        for expected_tool in expected_tools:
            assert (
                expected_tool in tool_names
            ), f"Expected tool {expected_tool} not found"
