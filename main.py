#!/usr/bin/env python3
"""
JARVIS Virtual Assistant - Main Entry Point.

This is the main entry point for the JARVIS-like virtual assistant.
It initializes the system, loads configuration, and starts the Brain
which orchestrates all agents.

Usage:
    python main.py
    python main.py --config /path/to/custom/settings.yaml
    python main.py --debug

Features:
    - Configuration loading from YAML
    - Environment variable overrides
    - Graceful shutdown handling
    - Debug mode for development

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                         Brain                                │
    │                    (Orchestrator)                            │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │                    Event Bus                             │ │
    │  │              (Pub/Sub Communication)                     │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │         ▲              ▲              ▲              ▲       │
    │         │              │              │              │       │
    │    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   │
    │    │  Voice  │   │ Intent  │   │ System  │   │ Memory  │   │
    │    │  Agent  │   │  Agent  │   │  Agent  │   │  Agent  │   │
    │    └─────────┘   └─────────┘   └─────────┘   └─────────┘   │
    └─────────────────────────────────────────────────────────────┘
    
    Flow:
    1. Voice Agent captures speech → VoiceInputEvent
    2. Intent Agent classifies → IntentRecognizedEvent
    3. System Agent executes → SystemCommandResultEvent
    4. Voice Agent speaks result ← VoiceOutputEvent
    5. Memory Agent tracks context throughout

TODO: Add CLI argument parsing with argparse
TODO: Add health check endpoint (HTTP)
TODO: Add configuration validation
TODO: Add plugin loading system
TODO: Add remote control API
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Priority:
    1. Specified config path
    2. JARVIS_CONFIG environment variable
    3. Default: config/settings.yaml
    
    Args:
        config_path: Optional path to config file
    
    Returns:
        Configuration dictionary
    
    TODO: Add environment variable overrides
    TODO: Add config schema validation
    """
    import yaml
    
    # Determine config path
    if config_path is None:
        config_path = os.environ.get(
            "JARVIS_CONFIG",
            str(PROJECT_ROOT / "config" / "settings.yaml")
        )
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        return {}
    
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        print(f"Loaded config from: {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    from utils.logger import configure_logging, init_from_config
    
    general_config = config.get("general", {})
    
    # Check for debug mode
    debug_mode = (
        general_config.get("debug_mode", False)
        or os.environ.get("JARVIS_DEBUG", "").lower() in ("1", "true", "yes")
        or "--debug" in sys.argv
    )
    
    if debug_mode:
        general_config["log_level"] = "DEBUG"
        print("Debug mode enabled")
    
    init_from_config(general_config)


def print_banner() -> None:
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                ║
    ║       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                ║
    ║       ██║███████║██████╔╝██║   ██║██║███████╗                ║
    ║  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                ║
    ║  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║                ║
    ║   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                ║
    ║                                                               ║
    ║           Virtual Assistant for macOS                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies() -> bool:
    """
    Check if required dependencies are available.
    
    Returns:
        True if all required dependencies are available
    
    TODO: Add more comprehensive dependency checking
    """
    missing = []
    
    # Check for PyYAML
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    # Check for optional but recommended packages
    optional_missing = []
    
    try:
        from google import genai
    except ImportError:
        optional_missing.append("google-genai (for Gemini-based intent recognition)")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    if optional_missing:
        print("\nOptional dependencies not installed:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print()
    
    return True


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dictionary of parsed arguments
    
    TODO: Use argparse for proper CLI handling
    """
    args = {
        "config": None,
        "debug": False,
    }
    
    argv = sys.argv[1:]
    i = 0
    
    while i < len(argv):
        arg = argv[i]
        
        if arg in ("--config", "-c") and i + 1 < len(argv):
            args["config"] = argv[i + 1]
            i += 2
        elif arg in ("--debug", "-d"):
            args["debug"] = True
            i += 1
        elif arg in ("--help", "-h"):
            print_help()
            sys.exit(0)
        else:
            i += 1
    
    return args


def print_help() -> None:
    """Print help message."""
    help_text = """
JARVIS Virtual Assistant

Usage:
    python main.py [options]

Options:
    -c, --config PATH   Path to configuration file
    -d, --debug         Enable debug mode
    -h, --help          Show this help message

Environment Variables:
    JARVIS_CONFIG       Path to configuration file
    JARVIS_DEBUG        Enable debug mode (1/true/yes)
    GEMINI_API_KEY      Gemini API key for intent recognition

Examples:
    python main.py
    python main.py --debug
    python main.py --config /path/to/settings.yaml
"""
    print(help_text)


async def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success)
    """
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Load configuration
    config = load_config(args.get("config"))
    
    # Override with command line args
    if args.get("debug"):
        config.setdefault("general", {})["debug_mode"] = True
    
    # Setup logging
    setup_logging(config)
    
    # Import after logging is configured
    from utils.logger import get_logger
    from orchestrator.brain import Brain
    
    logger = get_logger(__name__)
    logger.info("Starting JARVIS Virtual Assistant")
    
    # Create and run the brain
    brain = Brain(config=config)
    
    try:
        await brain.run()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
        
    finally:
        logger.info("JARVIS shutting down")


def run() -> None:
    """
    Entry point for running as a module.
    
    Can be invoked with: python -m jarvis
    """
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    run()
