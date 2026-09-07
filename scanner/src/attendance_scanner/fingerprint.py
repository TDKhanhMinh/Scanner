"""Fast file fingerprinting and streaming cryptographic hashing."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Union

from .contracts import BaseContract


def compute_fast_fingerprint(path: Union[str, Path]) -> Tuple[int, int]:
    """Retrieve file size and nanosecond modification time in a single stat call.

    Args:
        path: Path to the target file.

    Returns:
        Tuple of (size_in_bytes, mtime_in_nanoseconds).

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If stat call fails.
    """
    stat_res = os.stat(path)
    return stat_res.st_size, stat_res.st_mtime_ns


def compute_sha256(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute streaming SHA-256 hex digest without loading the whole file into RAM.

    Args:
        path: Path to the target file.
        chunk_size: Number of bytes to read per buffer chunk. Default 64KB. Must be > 0.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest string.

    Raises:
        ValueError: If chunk_size is less than or equal to 0.
        FileNotFoundError: If the file does not exist.
        OSError: If reading the file fails.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class FileFingerprint(BaseContract):
    """File metadata fingerprint with optional lazy SHA-256 digest."""

    path: str
    size: int
    mtime_ns: int
    sha256: Optional[str] = None

    def matches(self, other_size: int, other_mtime_ns: int) -> bool:
        """Fast check if size and nanosecond mtime match without reading disk."""
        return self.size == other_size and self.mtime_ns == other_mtime_ns

    def ensure_sha256(self) -> str:
        """Lazily compute and cache SHA-256 digest if not already present."""
        if self.sha256 is None:
            self.sha256 = compute_sha256(self.path)
        return self.sha256

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        compute_hash: bool = False,
    ) -> "FileFingerprint":
        """Construct a FileFingerprint by querying the filesystem.

        Args:
            path: Path to the file.
            compute_hash: If True, computes SHA-256 immediately; otherwise None.

        Returns:
            FileFingerprint instance.
        """
        p = Path(path)
        size, mtime_ns = compute_fast_fingerprint(p)
        digest = compute_sha256(p) if compute_hash else None
        return cls(
            path=str(p),
            size=size,
            mtime_ns=mtime_ns,
            sha256=digest,
        )
