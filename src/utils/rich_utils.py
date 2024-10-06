from pathlib import Path
from typing import Sequence, Any, Dict

import rich
import rich.syntax
import rich.tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
from rich.panel import Panel

console = Console()


def print_config_tree(cfg, resolve: bool = False, save_to_file: bool = False, log_dir: str = None):
    """Prints content of config using Rich library and its tree structure."""

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # Add all fields to queue
    for field in cfg:
        queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, dict):
            branch_content = str(config_group)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    console.print(tree)

    # save config tree to file
    if save_to_file:
        if log_dir is None:
            log_dir = Path.cwd() / "logs"  # Default to a 'logs' directory in the current working directory
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "config_tree.log"
        
        with open(log_file, "w") as file:
            rich.print(tree, file=file)

    return tree


def print_rich_progress(text: str):
    """Prints a rich progress bar with spinner."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(description=text, total=100)
        while not progress.finished:
            progress.update(task, advance=0.9)


def print_rich_panel(text: str, title: str):
    """Prints a rich panel with given text and title."""
    console.print(Panel(text, title=title))
