"""Typer CLI entry point for SciEmbed."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(name="sciembed", help="SciEmbed: Modern scientific embedding model.")
console = Console()

data_app = typer.Typer(help="Data pipeline commands.")
train_app = typer.Typer(help="Training commands.")
eval_app = typer.Typer(help="Evaluation commands.")

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")


@data_app.command("build-mlm-corpus")
def build_mlm_corpus(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/mlm_corpus.yaml"
    ),
    source: Annotated[Optional[str], typer.Option(help="Single source to process (for multi-node parallelism)")] = None,
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Build the MDS corpus for MLM pretraining."""
    from sciembed.config import load_typed_config, MLMCorpusConfig
    from sciembed.data.mlm_corpus import build_corpus

    cfg = load_typed_config(config, MLMCorpusConfig, overrides)
    if source:
        console.print(f"[bold]Building MLM corpus ({source}) → {cfg.output_dir}/{source}[/bold]")
    else:
        console.print(f"[bold]Building MLM corpus → {cfg.output_dir}[/bold]")
    build_corpus(cfg, single_source=source)


@data_app.command("build-triplets-prep")
def build_triplets_prep(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/citation_triplets.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Materialize shared data (paper pool + forward citations) for triplet generation."""
    from sciembed.config import load_typed_config, CitationTripletsConfig
    from sciembed.data.citation_triplets import prep_shared_data

    cfg = load_typed_config(config, CitationTripletsConfig, overrides)
    console.print(f"[bold]Preparing shared data → {cfg.output_dir}[/bold]")
    prep_shared_data(cfg)


@data_app.command("build-triplets")
def build_triplets(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/citation_triplets.yaml"
    ),
    worker_id: Annotated[Optional[int], typer.Option(help="Worker ID for multi-node parallelism")] = None,
    num_workers: Annotated[Optional[int], typer.Option(help="Total workers for multi-node parallelism")] = None,
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Generate citation triplets for contrastive training."""
    from sciembed.config import load_typed_config, CitationTripletsConfig
    from sciembed.data.citation_triplets import build_triplets as _build

    cfg = load_typed_config(config, CitationTripletsConfig, overrides)
    if worker_id is not None:
        console.print(f"[bold]Building triplets (worker {worker_id}/{num_workers}) → {cfg.output_dir}/worker_{worker_id}[/bold]")
    else:
        console.print(f"[bold]Building citation triplets → {cfg.output_dir}[/bold]")
    _build(cfg, worker_id=worker_id, num_workers=num_workers)


@data_app.command("build-context-pairs")
def build_context_pairs(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/citation_contexts.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Generate citation context → document pairs (Signal B)."""
    from sciembed.config import load_typed_config, CitationContextsConfig
    from sciembed.data.citation_contexts import build_context_pairs as _build

    cfg = load_typed_config(config, CitationContextsConfig, overrides)
    console.print(f"[bold]Building citation context pairs → {cfg.output_dir}[/bold]")
    _build(cfg)


@data_app.command("build-intent-pairs")
def build_intent_pairs(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/intent_triplets.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Generate intent-conditioned citation pairs (Signal C)."""
    from sciembed.config import load_typed_config, IntentTripletsConfig
    from sciembed.data.intent_triplets import build_intent_pairs as _build

    cfg = load_typed_config(config, IntentTripletsConfig, overrides)
    console.print(f"[bold]Building intent pairs → {cfg.output_dir}[/bold]")
    _build(cfg)


@data_app.command("build-section-pairs")
def build_section_pairs(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/section_pairs.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Generate section-aware training pairs (Signal E)."""
    from sciembed.config import load_typed_config, SectionPairsConfig
    from sciembed.data.section_extractor import build_section_pairs as _build

    cfg = load_typed_config(config, SectionPairsConfig, overrides)
    console.print(f"[bold]Building section pairs → {cfg.output_dir}[/bold]")
    _build(cfg)


@data_app.command("mix-signals")
def mix_signals(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/data_mixer.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Mix all training signals for Stage 2 contrastive training."""
    from sciembed.config import load_typed_config, DataMixerConfig
    from sciembed.data.data_mixer import mix_signals as _mix

    cfg = load_typed_config(config, DataMixerConfig, overrides)
    console.print(f"[bold]Mixing signals → {cfg.output_dir}[/bold]")
    _mix(cfg)


@data_app.command("build-instruction-pairs")
def build_instruction_pairs(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/data/instruction_pairs.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Generate instruction-aware training pairs."""
    from sciembed.config import load_typed_config, InstructionPairsConfig
    from sciembed.data.instruction_pairs import build_instruction_pairs as _build

    cfg = load_typed_config(config, InstructionPairsConfig, overrides)
    console.print(f"[bold]Building instruction pairs → {cfg.output_dir}[/bold]")
    _build(cfg)


@train_app.command("stage1")
def train_stage1(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/train/stage1_mlm.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Run Stage 1 domain-adaptive MLM pretraining."""
    from sciembed.config import load_typed_config, Stage1MLMConfig
    from sciembed.train.mlm import train_mlm

    cfg = load_typed_config(config, Stage1MLMConfig, overrides)
    console.print(f"[bold]Stage 1 MLM pretraining → {cfg.output_dir}[/bold]")
    train_mlm(cfg)


@train_app.command("stage2")
def train_stage2(
    config: Annotated[Path, typer.Option(help="Config YAML path")] = Path(
        "configs/train/stage2_contrastive.yaml"
    ),
    overrides: Annotated[Optional[list[str]], typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Run Stage 2 contrastive fine-tuning."""
    from sciembed.config import load_typed_config, Stage2ContrastiveConfig
    from sciembed.train.contrastive import train_contrastive

    cfg = load_typed_config(config, Stage2ContrastiveConfig, overrides)
    console.print(f"[bold]Stage 2 contrastive training → {cfg.output_dir}[/bold]")
    train_contrastive(cfg)


@train_app.command("convert-checkpoint")
def convert_checkpoint(
    composer_checkpoint: Annotated[Path, typer.Argument(help="Composer .pt checkpoint path")],
    output_dir: Annotated[Path, typer.Argument(help="HuggingFace output directory")],
    model_name: Annotated[str, typer.Option(help="Base model name")] = "answerdotai/ModernBERT-base",
) -> None:
    """Convert a Composer checkpoint to HuggingFace format."""
    from sciembed.train.model import convert_composer_to_hf

    console.print(f"[bold]Converting {composer_checkpoint} → {output_dir}[/bold]")
    convert_composer_to_hf(composer_checkpoint, output_dir, model_name)


# Evaluation deliberately has no subcommands here: we only ever report numbers
# from the upstream harnesses (allenai/scirepeval and the mteb package) so the
# results stay comparable to published baselines. See docs/REPRODUCTION.md.


if __name__ == "__main__":
    app()
