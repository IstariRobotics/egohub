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
        "--tool",
        type=str,
        required=True,
        help="The name of the tool class to apply (e.g., GradioHandTrackingTool).",
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
        spec = importlib.util.find_spec(tools_module_name)
        if spec is None or spec.origin is None:
            logger.error(f"Could not find the '{tools_module_name}' module.")
            return None
        tools_dir = Path(spec.origin).parent
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

    # Find and instantiate the selected tool
    tool_class = find_tool_class(args.tool)
    if not tool_class:
        logger.error(f"Tool class '{args.tool}' not found in egohub.tools.")
        return
        
    # Parse tool arguments
    tool_kwargs = {}
    if args.tool_args:
        for arg in args.tool_args:
            key, value = arg.split('=', 1)
            tool_kwargs[key] = value

    try:
        tool: BaseTool = tool_class(**tool_kwargs)
    except TypeError as e:
        logger.error(f"Failed to instantiate tool '{args.tool}' with arguments {tool_kwargs}. Error: {e}")
        return


    logger.info(f"Processing file: {args.input_file} with tool: {args.tool}")

    with h5py.File(args.input_file, "a") as f:
        trajectory_keys = sorted([key for key in f.keys() if key.startswith("trajectory_")])
        logger.info(f"Found {len(trajectory_keys)} trajectories to process.")

        for key in tqdm(trajectory_keys, desc="Processing Trajectories"):
            logger.info(f"--- Processing {key} ---")
            traj_group = f[key]
            if not isinstance(traj_group, h5py.Group):
                logger.warning(f"Skipping key '{key}' as it is not an HDF5 group.")
                continue
            tool(traj_group)

    logger.info("Processing complete.")


if __name__ == "__main__":
    main() 