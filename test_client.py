import asyncio
import base64
import json
import os
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def open_base64_png(b64_string: str):
    """
    Decodes a base64-encoded PNG and opens it using the default image viewer.

    Works in IPython terminal sessions.
    """
    if b64_string.startswith("data:image"):
        b64_string = b64_string.split(",", 1)[1]

    image_data = base64.b64decode(b64_string)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        f.write(image_data)
        tmp_path = f.name

    if os.uname().sysname == "Darwin":
        os.system(f'open "{tmp_path}"')
    else:
        os.system(f'xdg-open "{tmp_path}"')


async def main():
    server_params = StdioServerParameters(command="python", args=["mcp_ols.py"])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print("=" * 50)
                print(f"{tool.name}: {tool.description}")

            result = await session.call_tool("create_analysis_session")
            session_id = result.content[0].text

            result = await session.call_tool(
                "load_data",
                {"session_id": session_id, "file_path": "~/Documents/Advertising.csv"},
            )

            result = await session.call_tool(
                "describe_data", {"session_id": session_id}
            )
            print(result.content[0].text)

            result = await session.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ Newspaper + Radio"},
            )
            print(json.loads(result.content[0].text)["summary"])

            result = await session.call_tool(
                "run_ols_regression",
                {"session_id": session_id, "formula": "Sales ~ TV + Newspaper + Radio"},
            )
            print(json.loads(result.content[0].text)["summary"])

            result = await session.call_tool("list_models", {"session_id": session_id})
            model_infos = [json.loads(x.text) for x in result.content]

            result = await session.call_tool(
                "model_assumptions_test",
                {"session_id": session_id, "model_id": model_infos[-1]["model_id"]},
            )
            print(result.content[0].text)

            result = await session.call_tool(
                "influence_diagnostics",
                {"session_id": session_id, "model_id": model_infos[-1]["model_id"]},
            )
            assert not result.isError

            result = await session.call_tool(
                "compare_models",
                {
                    "session_id": session_id,
                    "model_ids": [model_info["model_id"] for model_info in model_infos],
                },
            )
            print(result.content[0].text)

            result = await session.call_tool(
                "visualize_model_comparison",
                {
                    "session_id": session_id,
                    "model_ids": [model_info["model_id"] for model_info in model_infos],
                },
            )
            assert not result.isError
            open_base64_png(result.content[0].data)

            result = await session.call_tool(
                "create_partial_dependence_plot",
                {
                    "session_id": session_id,
                    "model_id": model_infos[-1]["model_id"],
                    "feature": "TV",
                },
            )
            assert not result.isError
            open_base64_png(result.content[0].data)


if __name__ == "__main__":
    asyncio.run(main())
