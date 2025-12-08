"""
Main entry point for Federated SAM2 Medical Image Segmentation.

This script is executed by SyftBox to run the federated learning workflow.
It determines whether to run as:
- FL Client (Data Owner / Hospital)
- FL Server (Data Scientist / Aggregator)
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("SYFT-FLWR SAM2 Segmentation - main.py starting...")
print("=" * 60)

# Debug: Print environment variables
print(f"SYFTBOX_EMAIL: {os.getenv('SYFTBOX_EMAIL')}")
print(f"SYFTBOX_FOLDER: {os.getenv('SYFTBOX_FOLDER')}")
print(f"DATA_DIR: {os.getenv('DATA_DIR')}")
print(f"OUTPUT_DIR: {os.getenv('OUTPUT_DIR')}")

from syft_flwr.client import create_client
from syft_flwr.config import load_flwr_pyproject
from syft_flwr.run import syftbox_run_flwr_client, syftbox_run_flwr_server

DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")


flower_project_dir = Path(__file__).parent.absolute()
print(f"flower_project_dir: {flower_project_dir}")

print("Creating client...")
client = create_client(project_dir=flower_project_dir)
print(f"Client created: email={client.email}")
print(f"Client my_datasite: {client.my_datasite}")
print(f"Client datasites: {client.datasites}")

print("Loading FL config from pyproject.toml...")
config = load_flwr_pyproject(flower_project_dir)
print(f"Config datasites: {config['tool']['syft_flwr']['datasites']}")
print(f"Config aggregator: {config['tool']['syft_flwr']['aggregator']}")

is_client = client.email in config["tool"]["syft_flwr"]["datasites"]
is_server = client.email in config["tool"]["syft_flwr"]["aggregator"]

print(f"is_client (Data Owner): {is_client}")
print(f"is_server (Data Scientist): {is_server}")

if is_client:
    # Run by each DO (hospital/data owner)
    print("\n" + "=" * 60)
    print("Running as FL CLIENT (Data Owner / Hospital)")
    print("Training SAM2 LoRA adapters on local medical data...")
    print("=" * 60 + "\n")

    syftbox_run_flwr_client(flower_project_dir)

    print("\n FL CLIENT completed!")
    sys.exit(0)

elif is_server:
    # Run by the DS (data scientist / aggregator)
    print("\n" + "=" * 60)
    print("Running as FL SERVER (Data Scientist / Aggregator)")
    print("Coordinating federated learning and aggregating adapters...")
    print("=" * 60 + "\n")

    syftbox_run_flwr_server(flower_project_dir)

    print("\n FL SERVER completed!")
    sys.exit(0)

else:
    print(f"\n ERROR: {client.email} is not in config.datasites or config.aggregator")
    print("Please check pyproject.toml configuration.")
    raise ValueError(f"{client.email} is not in config.datasites or config.aggregator")
    sys.exit(1)
