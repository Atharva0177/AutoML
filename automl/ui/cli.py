"""Command-line interface for AutoML."""

import click
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from automl import AutoML
from automl.utils.logger import logger

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AutoML - Automated Machine Learning System"""
    pass


@cli.command()
@click.option("--input", "-i", required=True, help="Path to input data file")
@click.option("--target", "-t", help="Target column name")
@click.option("--output", "-o", default="./results", help="Output directory")
@click.option("--validate-only", is_flag=True, help="Only validate data without training")
def train(input: str, target: str, output: str, validate_only: bool):
    """
    Train AutoML models on your dataset.
    
    Example:
        automl train -i data.csv -t target_column -o results/
    """
    try:
        console.print("[bold blue]AutoML Training Pipeline[/bold blue]")
        console.print(f"Input file: {input}")
        console.print(f"Target column: {target or 'Not specified'}")
        console.print(f"Output directory: {output}")
        console.print()
        
        # Initialize AutoML
        with console.status("[bold green]Initializing AutoML..."):
            aml = AutoML()
        
        # Load data
        with console.status("[bold green]Loading data..."):
            aml.load_data(input, target_column=target)
        
        # Display data info
        console.print("[bold green]✓[/bold green] Data loaded successfully")
        info = aml.get_data_info()
        
        table = Table(title="Dataset Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Shape", f"{info['shape'][0]} rows × {info['shape'][1]} columns")
        table.add_row("Target Column", str(info.get('target_column', 'Not set')))
        table.add_row("Quality Score", f"{info.get('quality_score', 0):.1f}/100")
        table.add_row("Missing Values", f"{info.get('missing_percentage', 0):.1f}%")
        
        console.print(table)
        
        # Save metadata
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_path = output_path / "metadata.json"
        aml.save_metadata(metadata_path)
        console.print(f"\n[green]Metadata saved to {metadata_path}[/green]")
        
        if validate_only:
            console.print("\n[yellow]Validation only mode - skipping training[/yellow]")
            return
        
        # Training (placeholder)
        console.print("\n[yellow]Note: Model training will be implemented in Phase 1, Month 3[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        logger.exception("Training failed")
        raise click.Abort()


@cli.command()
@click.option("--path", "-p", required=True, help="Path to results directory")
def results(path: str):
    """
    View results from a previous training run.
    
    Example:
        automl results -p results/
    """
    try:
        results_path = Path(path)
        metadata_path = results_path / "metadata.json"
        
        if not metadata_path.exists():
            console.print(f"[red]Error: No metadata found in {path}[/red]")
            return
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Display results
        console.print("[bold blue]AutoML Results[/bold blue]\n")
        
        # File info
        if "file" in metadata:
            table = Table(title="File Information")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")
            
            file_info = metadata["file"]
            table.add_row("File Path", file_info.get("filepath", "N/A"))
            table.add_row("File Size", file_info.get("file_size_formatted", "N/A"))
            table.add_row("Rows", str(file_info.get("n_rows", "N/A")))
            table.add_row("Columns", str(file_info.get("n_columns", "N/A")))
            
            console.print(table)
            console.print()
        
        # Quality report
        if "quality" in metadata:
            quality = metadata["quality"]
            console.print(f"[bold]Quality Score:[/bold] {quality.get('overall_score', 0):.1f}/100")
            
            if quality.get("recommendations"):
                console.print("\n[bold]Recommendations:[/bold]")
                for i, rec in enumerate(quality["recommendations"], 1):
                    console.print(f"  {i}. {rec}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
def info():
    """Display AutoML system information."""
    from automl import __version__
    
    console.print("[bold blue]AutoML System Information[/bold blue]\n")
    console.print(f"Version: {__version__}")
    console.print(f"Status: MVP (Phase 1)")
    console.print("\n[bold]Available Commands:[/bold]")
    console.print("  train    - Train models on your dataset")
    console.print("  results  - View training results")
    console.print("  info     - Display this information")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
