#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utilities for the financial ML pipeline.
"""
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def log_info(message: str) -> None:
    """Log an info message in green."""
    print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")


def log_warning(message: str) -> None:
    """Log a warning message in yellow."""
    print(f"{Fore.YELLOW}WARNING: {message}{Style.RESET_ALL}")


def log_error(message: str) -> None:
    """Log an error message in red."""
    print(f"{Fore.RED}ERROR: {message}{Style.RESET_ALL}")


def log_section(title: str) -> None:
    """Log a section header with decorative formatting."""
    separator = "=" * 60
    print(f"\n{Fore.CYAN}{separator}")
    print(f"{Fore.CYAN}{title}")
    print(f"{Fore.CYAN}{separator}{Style.RESET_ALL}")


def log_subsection(title: str) -> None:
    """Log a subsection header with lighter formatting."""
    separator = "-" * 40
    print(f"\n{Fore.YELLOW}{separator}")
    print(f"{Fore.YELLOW}{title}")
    print(f"{Fore.YELLOW}{separator}{Style.RESET_ALL}") 