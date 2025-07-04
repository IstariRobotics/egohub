"""Command-line script to apply processing tools to an HDF5 dataset."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path
import logging

import h5py
from tqdm import tqdm

from egohub.tools.base import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply processing tools to an HDF5 dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the input HDF5 file.",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs='+',
        required=True,
        help="One or more tool classes to apply in sequence (e.g., HuggingFaceObjectDetectionTool).",
    )
    parser.add_argument(
        '--tool-args',
        nargs='*',
        help="""Additional keyword arguments to pass to the tool's constructor,
in 'key=value' format. For example: model_src='user/model'""",
    )
    return parser.parse_args()


def find_tool_class(tool_name: str) -> type[BaseTool] | None:
    """Finds a tool class by name by scanning the egohub.tools module."""
    tools_module_name = "egohub.tools"
    
    # Get the path to the 'egohub/tools' directory
    try:
        # Find the path to the 'egohub' package
        egohub_spec = importlib.util.find_spec("egohub")
        if egohub_spec is None or egohub_spec.origin is None:
            logger.error("Could not find the 'egohub' package.")
            return None
        
        # Construct the path to the 'tools' subdirectory
        tools_dir = Path(egohub_spec.origin).parent / "tools"

        if not tools_dir.exists():
            logger.error(f"Tools directory not found at: {tools_dir}")
            return None
            
    except Exception as e:
        logger.error(f"Error finding tools directory: {e}")
        return None

    # Iterate over all python files in the tools directory
    for f in tools_dir.iterdir():
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("_"):
            module_short_name = f.stem
            try:
                module = importlib.import_module(f".{module_short_name}", tools_module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if name == tool_name and issubclass(obj, BaseTool) and obj is not BaseTool:
                        return obj
            except ImportError as e:
                logger.warning(f"Could not import or inspect module {module_short_name}: {e}")
                continue
    return None


def main() -> None:
    """Main entry point for the script."""
    args = get_args()

    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return

    # --- Parse tool arguments ---
    # This is a bit complex to allow per-tool arguments.
    # The command line would look like:
    # --tools Tool1 Tool2 --tool-args Tool1:arg1=val1 Tool2:arg1=val2 Tool2:arg2=val3
    tool_specific_args = {}
    if args.tool_args:
        for arg in args.tool_args:
            try:
                tool_name, arg_str = arg.split(':', 1)
                key, value = arg_str.split('=', 1)
                if tool_name not in tool_specific_args:
                    tool_specific_args[tool_name] = {}
                tool_specific_args[tool_name][key] = value
            except ValueError:
                logger.error(
                    f"Invalid tool argument format: '{arg}'. "
                    "Expected format: 'ToolName:key=value'"
                )
                return
                
    # --- Instantiate all tools first ---
    tools: list[BaseTool] = []
    for tool_name in args.tools:
        tool_class = find_tool_class(tool_name)
        if not tool_class:
            logger.error(f"Tool class '{tool_name}' not found in egohub.tools.")
            return

        kwargs = tool_specific_args.get(tool_name, {})
        try:
            tool_instance = tool_class(**kwargs)
            tools.append(tool_instance)
            logger.info(f"Successfully instantiated tool '{tool_name}' with args: {kwargs}")
        except TypeError as e:
            logger.error(f"Failed to instantiate tool '{tool_name}' with arguments {kwargs}. Error: {e}")
            return

    # --- Process the file with the pipeline of tools ---
    logger.info(f"Processing file: {args.input_file} with pipeline: {[t.__class__.__name__ for t in tools]}")

    with h5py.File(args.input_file, "a") as f:
        trajectory_keys = sorted([key for key in f.keys() if key.startswith("trajectory_")])
        logger.info(f"Found {len(trajectory_keys)} trajectories to process.")

        for key in tqdm(trajectory_keys, desc="Processing Trajectories"):
            logger.info(f"--- Processing {key} ---")
            traj_group = f[key]
            if not isinstance(traj_group, h5py.Group):
                logger.warning(f"Skipping key '{key}' as it is not an HDF5 group.")
                continue

            # Apply each tool in sequence to the same trajectory group
            for tool in tools:
                 logger.info(f"Applying tool: {tool.__class__.__name__}")
                 tool(traj_group)

    logger.info("Processing complete.")


if __name__ == "__main__":
    main() 