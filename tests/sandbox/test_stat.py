from __future__ import annotations

import io
from pathlib import Path

import pytest

from agents.sandbox.errors import ExecNonZeroError, InvalidManifestPathError
from agents.sandbox.files import EntryKind, FileEntry
from agents.sandbox.manifest import Manifest
from agents.sandbox.session import (
    CallbackSink,
    Instrumentation,
    SandboxSession,
    SandboxSessionEvent,
    SandboxSessionState,
)
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.snapshot import NoopSnapshot
from agents.sandbox.types import ExecResult, Permissions, User
from agents.sandbox.workspace_paths import SandboxPathGrant


class _StatSession(BaseSandboxSession):
    def __init__(self, entry: FileEntry) -> None:
        self.state = SandboxSessionState(
            type="test-stat",
            manifest=Manifest(
                root="/workspace",
                extra_path_grants=(
                    SandboxPathGrant(path="/missing-grant"),
                    SandboxPathGrant(path="/denied-grant"),
                    SandboxPathGrant(path="/invalid-grant"),
                ),
            ),
            snapshot=NoopSnapshot(id="test-stat"),
        )
        self.entry = entry
        self.exec_calls: list[tuple[str, ...]] = []
        self.stat_calls: list[tuple[Path | str, str | User | None]] = []
        self.validation_calls: list[Path | str] = []

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        _ = for_write
        self.validation_calls.append(path)
        if path == "escape.txt" or path == Path("/invalid-grant"):
            raise InvalidManifestPathError(rel=path, reason="escape_root")
        return self.normalize_path(path)

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        _ = timeout
        rendered = tuple(str(part) for part in command)
        self.exec_calls.append(rendered)
        target = rendered[-1]
        if target in {"/missing-grant", "/workspace/missing.txt"}:
            return ExecResult(stdout=b"", stderr=b"No such file or directory", exit_code=2)
        if target == "/denied-grant":
            return ExecResult(stdout=b"", stderr=b"permission denied", exit_code=13)
        if target == "/workspace":
            return ExecResult(
                stdout=b"drwxr-x--- 2 runner runner 64 Jan 1 00:00 /workspace\n",
                stderr=b"",
                exit_code=0,
            )
        if target == "/workspace/link -> name":
            return ExecResult(
                stdout=(
                    b"lrwxrwxrwx 1 runner runner 12 Jan 1 00:00 "
                    b"/workspace/link -> name -> /workspace/notes.txt\n"
                ),
                stderr=b"",
                exit_code=0,
            )
        return ExecResult(
            stdout=f"-rw-r----- 1 runner runner 12 Jan 1 00:00 {target}\n".encode(),
            stderr=b"",
            exit_code=0,
        )

    async def stat(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> FileEntry | None:
        self.stat_calls.append((path, user))
        return await super().stat(path, user=user)

    async def read(self, path: Path, *, user: str | User | None = None) -> io.IOBase:
        _ = (path, user)
        raise AssertionError("not expected")

    async def write(
        self,
        path: Path,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        _ = (path, data, user)
        raise AssertionError("not expected")

    async def running(self) -> bool:
        return True

    async def persist_workspace(self) -> io.IOBase:
        raise AssertionError("not expected")

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        _ = data
        raise AssertionError("not expected")


def _file_entry() -> FileEntry:
    return FileEntry(
        path="/workspace/notes.txt",
        permissions=Permissions.from_mode(0o100640),
        owner="runner",
        group="runner",
        size=12,
        kind=EntryKind.FILE,
    )


@pytest.mark.asyncio
async def test_base_stat_uses_portable_listing_fallback() -> None:
    entry = _file_entry()
    session = _StatSession(entry)

    result = await BaseSandboxSession.stat(session, "notes.txt", user="runner")

    assert result == entry
    assert len(session.exec_calls) == 2
    assert session.exec_calls[0][:8] == (
        "sudo",
        "-u",
        "runner",
        "--",
        "env",
        "LC_ALL=C",
        "ls",
        "-ld",
    )
    assert "python3" not in session.exec_calls[0]

    assert await BaseSandboxSession.stat(session, "missing.txt", user="runner") is None
    arrow_link = await BaseSandboxSession.stat(session, "link -> name", user="runner")
    assert arrow_link is not None
    assert arrow_link.path == "/workspace/link -> name"
    assert arrow_link.kind is EntryKind.SYMLINK
    assert await BaseSandboxSession.stat(session, "/missing-grant", user="runner") is None
    with pytest.raises(InvalidManifestPathError):
        await BaseSandboxSession.stat(session, "escape.txt", user="runner")
    with pytest.raises(ExecNonZeroError, match="permission denied"):
        await BaseSandboxSession.stat(session, "/denied-grant", user="runner")
    exec_call_count = len(session.exec_calls)
    with pytest.raises(InvalidManifestPathError):
        await BaseSandboxSession.stat(session, "/invalid-grant", user="runner")
    assert len(session.exec_calls) == exec_call_count
    assert session.validation_calls == [
        Path("/workspace"),
        "notes.txt",
        Path("/workspace"),
        "missing.txt",
        Path("/workspace"),
        "link -> name",
        Path("/missing-grant"),
        "/missing-grant",
        Path("/workspace"),
        "escape.txt",
        Path("/denied-grant"),
        Path("/invalid-grant"),
    ]


@pytest.mark.asyncio
async def test_sandbox_session_stat_delegates_and_emits_events() -> None:
    entry = _file_entry()
    inner = _StatSession(entry)
    events: list[SandboxSessionEvent] = []
    wrapped = SandboxSession(
        inner,
        instrumentation=Instrumentation(
            sinks=[CallbackSink(lambda event, _session: events.append(event), mode="sync")]
        ),
    )
    user = User(name="runner")

    result = await wrapped.stat(Path("notes.txt"), user=user)

    assert result == entry
    assert inner.stat_calls == [(Path("notes.txt"), user)]
    assert [(event.op, event.phase) for event in events] == [
        ("stat", "start"),
        ("stat", "finish"),
    ]
    assert events[0].data == {"path": "notes.txt", "user": "runner"}
