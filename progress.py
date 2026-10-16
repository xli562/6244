"""Terminal progress shared by collection, training, and evaluation commands."""

from tqdm import tqdm


def progress_bar(iterable=None, *, disabled:bool=False, **options) -> tqdm:
    """Return a bar with counts, elapsed time, rate, and ETA; hide it outside terminals."""
    return tqdm(iterable, disable=True if disabled else None, dynamic_ncols=True,
                mininterval=.2, **options)


def add_progress_arguments(parser) -> None:
    """Add common terminal and detailed-log options to an argument parser."""
    parser.add_argument('--no-progress', action='store_true', help='Disable terminal progress bars.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-trial or per-epoch results, including in redirected logs.')
