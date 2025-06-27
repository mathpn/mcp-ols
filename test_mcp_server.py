import json
import os
import tempfile

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            assert not result.isError
            session_id = result.content[0].text
            assert isinstance(session_id, str)
            assert len(session_id) > 0

            result2 = await session.call_tool("create_analysis_session")
            assert not result2.isError
            session_id2 = result2.content[0].text
            assert session_id != session_id2


@pytest.mark.asyncio
async def test_data_loading(sample_csv_data):
    """Test loading data from CSV files"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(sample_csv_data)
                temp_path = f.name

            try:
                result = await session.call_tool(
                    "load_data", {"session_id": session_id, "file_path": temp_path}
                )
                assert not result.isError

                result = await session.call_tool(
                    "describe_data", {"session_id": session_id}
                )
                assert not result.isError
                description = result.content[0].text
                assert "TV" in description
                assert "Radio" in description
                assert "Sales" in description
            finally:
                os.unlink(temp_path)


@pytest.mark.asyncio
async def test_data_loading_errors():
    """Test error handling in data loading"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            result = await session.call_tool(
                "load_data",
                {"session_id": session_id, "file_path": "/nonexistent/file.csv"},
            )
            assert result.isError

            result = await session.call_tool(
                "load_data",
                {"session_id": "invalid-session", "file_path": "/tmp/file.csv"},
            )
            assert result.isError


@pytest.mark.asyncio
async def test_ols_regression(sample_csv_data):
    """Test OLS regression functionality"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(sample_csv_data)
                temp_path = f.name

            try:
                await session.call_tool(
                    "load_data", {"session_id": session_id, "file_path": temp_path}
                )

                result = await session.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
                )
                assert not result.isError
                model_info = json.loads(result.content[0].text)
                assert "model_id" in model_info
                assert "summary" in model_info
                assert "TV" in model_info["summary"]
                assert "Radio" in model_info["summary"]

                result2 = await session.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "Sales ~ TV"},
                )
                assert not result2.isError
                model_info2 = json.loads(result2.content[0].text)
                assert model_info["model_id"] != model_info2["model_id"]

                result3 = await session.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "NonexistentColumn ~ TV"},
                )
                assert result3.isError

            finally:
                os.unlink(temp_path)


@pytest.mark.asyncio
async def test_logistic_regression(sample_logistic_data):
    """Test logistic regression functionality"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(sample_logistic_data)
                temp_path = f.name

            try:
                await session.call_tool(
                    "load_data", {"session_id": session_id, "file_path": temp_path}
                )

                result = await session.call_tool(
                    "run_logistic_regression",
                    {
                        "session_id": session_id,
                        "formula": "passed ~ hours_studied + practice_exams",
                    },
                )
                assert not result.isError
                model_info = json.loads(result.content[0].text)
                assert "model_id" in model_info
                assert "summary" in model_info

            finally:
                os.unlink(temp_path)


@pytest.mark.asyncio
async def test_model_diagnostics(sample_csv_data):
    """Test model diagnostic functionality"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(sample_csv_data)
                temp_path = f.name

            try:
                await session.call_tool(
                    "load_data", {"session_id": session_id, "file_path": temp_path}
                )

                result = await session.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
                )
                model_info = json.loads(result.content[0].text)
                model_id = model_info["model_id"]

                result = await session.call_tool(
                    "create_residual_plots",
                    {"session_id": session_id, "model_id": model_id},
                )
                assert not result.isError
                # The server returns raw base64 image data
                image_data = result.content[0].data
                assert isinstance(image_data, str) and len(image_data) > 0

                result = await session.call_tool(
                    "model_assumptions_test",
                    {"session_id": session_id, "model_id": model_id},
                )
                assert not result.isError
                test_results = result.content[0].text
                assert "Jarque-Bera" in test_results
                assert "Breusch-Pagan" in test_results

                result = await session.call_tool(
                    "influence_diagnostics",
                    {"session_id": session_id, "model_id": model_id},
                )
                assert not result.isError
                image_data = result.content[0].data
                assert isinstance(image_data, str) and len(image_data) > 0

                result = await session.call_tool(
                    "vif_table", {"session_id": session_id, "model_id": model_id}
                )
                assert not result.isError
                vif_data = result.content[0].text
                assert "VIF" in vif_data

            finally:
                os.unlink(temp_path)


@pytest.mark.asyncio
async def test_model_comparison(sample_csv_data):
    """Test model comparison functionality"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(sample_csv_data)
                temp_path = f.name

            try:
                await session.call_tool(
                    "load_data", {"session_id": session_id, "file_path": temp_path}
                )

                result1 = await session.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "Sales ~ TV"},
                )
                model1_id = json.loads(result1.content[0].text)["model_id"]

                result2 = await session.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
                )
                model2_id = json.loads(result2.content[0].text)["model_id"]

                result = await session.call_tool(
                    "list_models", {"session_id": session_id}
                )
                assert not result.isError
                models = [json.loads(content.text) for content in result.content]
                assert len(models) >= 2
                model_ids = [model["model_id"] for model in models]
                assert model1_id in model_ids
                assert model2_id in model_ids

                result = await session.call_tool(
                    "compare_models",
                    {"session_id": session_id, "model_ids": [model1_id, model2_id]},
                )
                assert not result.isError
                comparison = result.content[0].text
                assert "AIC" in comparison
                assert "BIC" in comparison

                result = await session.call_tool(
                    "visualize_model_comparison",
                    {"session_id": session_id, "model_ids": [model1_id, model2_id]},
                )
                assert not result.isError
                image_data = result.content[0].data
                assert isinstance(image_data, str) and len(image_data) > 0

            finally:
                os.unlink(temp_path)


@pytest.mark.asyncio
async def test_partial_dependence_plots(sample_csv_data):
    """Test partial dependence plot functionality"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(sample_csv_data)
                temp_path = f.name

            try:
                await session.call_tool(
                    "load_data", {"session_id": session_id, "file_path": temp_path}
                )

                result = await session.call_tool(
                    "run_ols_regression",
                    {"session_id": session_id, "formula": "Sales ~ TV + Radio"},
                )
                model_info = json.loads(result.content[0].text)
                model_id = model_info["model_id"]

                result = await session.call_tool(
                    "create_partial_dependence_plot",
                    {
                        "session_id": session_id,
                        "model_id": model_id,
                        "feature": "TV",
                    },
                )
                assert not result.isError
                image_data = result.content[0].data
                assert isinstance(image_data, str) and len(image_data) > 0

                result = await session.call_tool(
                    "create_partial_dependence_plot",
                    {
                        "session_id": session_id,
                        "model_id": model_id,
                        "feature": "NonexistentFeature",
                    },
                )
                assert result.isError

            finally:
                os.unlink(temp_path)


@pytest.mark.asyncio
async def test_tool_listing():
    """Test that all expected tools are available"""
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.list_tools()
            tool_names = [tool.name for tool in result.tools]

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
                assert expected_tool in tool_names, (
                    f"Expected tool {expected_tool} not found"
                )
