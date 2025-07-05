"""Typed helper protocols representing common HDF5 group layouts.

These are *structural* (``Protocol``-based) interfaces that describe only the
parts of ``h5py.Group`` we rely on. They allow static-type checkers to validate
string-key access without us needing a concrete subclass of ``h5py.Group``.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import h5py


@runtime_checkable
class H5GroupLike(Protocol):
    """Minimal subset of ``h5py.Group`` behaviour we use."""

    def __getitem__(self, key: str) -> h5py.Dataset | h5py.Group:  # noqa: D401
        ...

    def __contains__(self, key: str) -> bool:  # noqa: D401
        ...

    def keys(self) -> Iterator[str]:  # noqa: D401
        ...


class CameraGroup(H5GroupLike, Protocol):
    """Expected layout for a camera sub-group."""

    def __getitem__(self, key: str) -> h5py.Dataset:  # type: ignore[override]
        ...

    # Datasets that must exist (names only for static analysis)
    pose_in_world: h5py.Dataset  # type: ignore[assignment]
    intrinsics: h5py.Dataset  # type: ignore[assignment]


class TrajectoryGroup(H5GroupLike, Protocol):
    """Expected layout for a top-level trajectory group."""

    def __getitem__(self, key: str) -> h5py.Group | h5py.Dataset:  # type: ignore[override]
        ...

    cameras: h5py.Group  # type: ignore[assignment]
    metadata: h5py.Group  # type: ignore[assignment]
