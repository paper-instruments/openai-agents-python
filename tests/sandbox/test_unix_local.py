from __future__ import annotations

import asyncio
import os
import signal
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agents.sandbox.errors import PtySessionNotFoundError
from agents.sandbox.manifest import Manifest
from agents.sandbox.sandboxes.unix_local import (
    UnixLocalSandboxClient,
    UnixLocalSandboxSession,
    UnixLocalSandboxSessionState,
    _UnixPtyProcessEntry,
)
from agents.sandbox.session.pty_types import PTY_PROCESSES_MAX
from agents.sandbox.snapshot import NoopSnapshot
from agents.sandbox.types import ExecResult, User


class _RecordingUnixLocalSession(UnixLocalSandboxSession):
    def __init__(self, root: Path) -> None:
        super().__init__(
            state=UnixLocalSandboxSessionState(
                manifest=Manifest(root=str(root)),
                snapshot=NoopSnapshot(id="noop"),
            )
        )
        self.exec_commands: list[tuple[str, ...]] = []

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        _ = timeout
        self.exec_commands.append(tuple(str(part) for part in command))
        return ExecResult(stdout=b"", stderr=b"", exit_code=0)


class TestUnixLocalPty:
    @pytest.mark.asyncio
    async def test_tty_fd_close_is_owned_without_blocking_termination(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = _RecordingUnixLocalSession(tmp_path)
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        async def blocked_to_thread(*args: object, **kwargs: object) -> None:
            _ = (args, kwargs)
            close_started.set()
            await release_close.wait()

        monkeypatch.setattr(asyncio, "to_thread", blocked_to_thread)
        process = cast(
            asyncio.subprocess.Process,
            SimpleNamespace(returncode=0, pid=None),
        )
        entry = _UnixPtyProcessEntry(process=process, tty=True, primary_fd=123)

        await asyncio.wait_for(session._terminate_pty_entry(entry), timeout=0.5)
        await close_started.wait()

        assert len(session._fd_close_tasks) == 1
        await asyncio.wait_for(session._after_stop(), timeout=0.5)
        assert len(session._fd_close_tasks) == 1

        release_close.set()
        await asyncio.gather(*session._fd_close_tasks)
        await asyncio.sleep(0)

        assert session._fd_close_tasks == set()

    @pytest.mark.asyncio
    async def test_pty_exec_write_poll_and_unknown_session_errors(self, tmp_path: Path) -> None:
        client = UnixLocalSandboxClient()
        manifest = Manifest(root=str(tmp_path / "workspace"))

        async with await client.create(manifest=manifest, snapshot=None, options=None) as session:
            started = await session.pty_exec_start(
                "sh",
                "-c",
                "IFS= read -r line; printf '%s\\n' \"$line\"",
                shell=False,
                tty=True,
                yield_time_s=0.05,
            )

            assert started.process_id is not None
            assert started.exit_code is None

            written = await session.pty_write_stdin(
                session_id=started.process_id,
                chars="hello from pty\n",
                yield_time_s=0.25,
            )
            assert written.process_id is None
            assert written.exit_code == 0
            assert "hello from pty" in written.output.decode("utf-8", errors="replace")

            with pytest.raises(PtySessionNotFoundError):
                await session.pty_write_stdin(session_id=started.process_id, chars="")

            with pytest.raises(PtySessionNotFoundError):
                await session.pty_write_stdin(session_id=999_999, chars="")

    @pytest.mark.asyncio
    async def test_pty_ctrl_c_interrupts_long_running_process(self, tmp_path: Path) -> None:
        client = UnixLocalSandboxClient()
        manifest = Manifest(root=str(tmp_path / "workspace"))

        async with await client.create(manifest=manifest, snapshot=None, options=None) as session:
            started = await session.pty_exec_start(
                "sleep",
                "30",
                shell=False,
                tty=True,
                yield_time_s=0.05,
            )

            assert started.process_id is not None
            assert started.exit_code is None

            first_interrupt = await session.pty_write_stdin(
                session_id=started.process_id,
                chars="\x03",
                yield_time_s=0.25,
            )
            if first_interrupt.process_id is None:
                interrupted = first_interrupt
            else:
                interrupted = await session.pty_write_stdin(
                    session_id=started.process_id,
                    chars="",
                    yield_time_s=5.5,
                )

            assert interrupted.process_id is None
            assert interrupted.exit_code is not None

            with pytest.raises(PtySessionNotFoundError):
                await session.pty_write_stdin(session_id=started.process_id, chars="")

    @pytest.mark.parametrize(
        ("signum", "chars"),
        [
            pytest.param(signal.SIGINT, "\x03", id="sigint"),
            pytest.param(signal.SIGQUIT, "\x1c", id="sigquit"),
        ],
    )
    @pytest.mark.asyncio
    async def test_pty_terminal_signals_interrupt_even_if_parent_ignores_signal(
        self, tmp_path: Path, signum: signal.Signals, chars: str
    ) -> None:
        client = UnixLocalSandboxClient()
        manifest = Manifest(root=str(tmp_path / "workspace"))
        previous_handler = signal.getsignal(signum)

        signal.signal(signum, signal.SIG_IGN)
        try:
            async with await client.create(
                manifest=manifest, snapshot=None, options=None
            ) as session:
                started = await session.pty_exec_start(
                    "sleep",
                    "30",
                    shell=False,
                    tty=True,
                    yield_time_s=0.05,
                )
                assert started.process_id is not None

                interrupted = await session.pty_write_stdin(
                    session_id=started.process_id,
                    chars=chars,
                    yield_time_s=5.5,
                )

                assert interrupted.process_id is None
                assert interrupted.exit_code == -signum
        finally:
            signal.signal(signum, previous_handler)

    @pytest.mark.asyncio
    async def test_non_tty_pty_session_rejects_stdin_and_can_still_be_polled(
        self, tmp_path: Path
    ) -> None:
        client = UnixLocalSandboxClient()
        manifest = Manifest(root=str(tmp_path / "workspace"))

        async with await client.create(manifest=manifest, snapshot=None, options=None) as session:
            started = await session.pty_exec_start(
                "sh",
                "-c",
                "printf 'stdout\\n'; printf 'stderr\\n' >&2; sleep 1",
                shell=False,
                tty=False,
                yield_time_s=0.05,
            )

            assert started.process_id is not None
            assert started.exit_code is None
            started_text = started.output.decode("utf-8", errors="replace")
            assert "stdout" in started_text
            assert "stderr" in started_text

            with pytest.raises(RuntimeError, match="stdin is not available for this process"):
                await session.pty_write_stdin(session_id=started.process_id, chars="hello")

            finished = await session.pty_write_stdin(
                session_id=started.process_id,
                chars="",
                yield_time_s=5.5,
            )
            text = finished.output.decode("utf-8", errors="replace")
            assert finished.process_id is None
            assert finished.exit_code == 0
            assert text == ""

            with pytest.raises(PtySessionNotFoundError):
                await session.pty_write_stdin(session_id=started.process_id, chars="")

    @pytest.mark.asyncio
    async def test_stop_terminates_active_pty_sessions(self, tmp_path: Path) -> None:
        client = UnixLocalSandboxClient()
        manifest = Manifest(root=str(tmp_path / "workspace"))

        session = await client.create(manifest=manifest, snapshot=None, options=None)
        await session.start()
        started = await session.pty_exec_start(
            "sh",
            "-c",
            "printf 'ready\\n'; sleep 30",
            shell=False,
            tty=True,
            yield_time_s=0.25,
        )

        assert started.process_id is not None
        assert "ready" in started.output.decode("utf-8", errors="replace")

        await session.stop()

        with pytest.raises(PtySessionNotFoundError):
            await session.pty_write_stdin(session_id=started.process_id, chars="")

    @pytest.mark.asyncio
    async def test_pty_terminate_stops_only_requested_session(self, tmp_path: Path) -> None:
        client = UnixLocalSandboxClient()
        manifest = Manifest(root=str(tmp_path / "workspace"))

        async with await client.create(manifest=manifest, snapshot=None, options=None) as session:
            terminated_process = await session.pty_exec_start(
                "sleep",
                "30",
                shell=False,
                tty=False,
                yield_time_s=0.05,
            )
            interactive_process = await session.pty_exec_start(
                "sh",
                "-c",
                "IFS= read -r line; printf '%s\\n' \"$line\"",
                shell=False,
                tty=True,
                yield_time_s=0.05,
            )

            assert terminated_process.process_id is not None
            assert interactive_process.process_id is not None

            terminated = await session.pty_terminate(terminated_process.process_id)

            assert terminated.process_id is None
            assert terminated.exit_code is not None

            interactive_finished = await session.pty_write_stdin(
                session_id=interactive_process.process_id,
                chars="still running\n",
                yield_time_s=0.25,
            )
            assert interactive_finished.process_id is None
            assert interactive_finished.exit_code == 0
            assert "still running" in interactive_finished.output.decode("utf-8", errors="replace")

            with pytest.raises(PtySessionNotFoundError):
                await session.pty_terminate(terminated_process.process_id)

    @pytest.mark.asyncio
    async def test_pty_terminate_drains_buffered_non_tty_output(self, tmp_path: Path) -> None:
        session = _RecordingUnixLocalSession(tmp_path)
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "printf 'buffered stdout'; printf 'buffered stderr' >&2; sleep 30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        try:
            # Leave the emitted bytes in the OS pipes until termination begins. The pump tasks
            # are intentionally created immediately before cleanup so cancelling them first
            # would deterministically lose the buffered output.
            await asyncio.sleep(0.05)
            entry = _UnixPtyProcessEntry(process=process, tty=False)
            entry.pump_tasks = [
                asyncio.create_task(session._pump_process_stream(entry, process.stdout)),
                asyncio.create_task(session._pump_process_stream(entry, process.stderr)),
            ]
            entry.wait_task = asyncio.create_task(session._watch_process_exit(entry))

            await session._terminate_pty_entry(entry)
            output, original_token_count = await session._collect_pty_output(
                entry=entry,
                yield_time_ms=0,
                max_output_tokens=None,
            )

            assert original_token_count is None
            assert b"buffered stdout" in output
            assert b"buffered stderr" in output
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    @pytest.mark.asyncio
    async def test_pty_terminate_wakes_in_flight_empty_poll(self, tmp_path: Path) -> None:
        session = _RecordingUnixLocalSession(tmp_path)
        try:
            started = await session.pty_exec_start(
                "sleep",
                "30",
                shell=False,
                tty=False,
                yield_time_s=0.05,
            )
            assert started.process_id is not None
            entry = session._pty_processes[started.process_id]

            poll_task = asyncio.create_task(
                session.pty_write_stdin(
                    session_id=started.process_id,
                    chars="",
                    yield_time_s=30,
                )
            )
            for _ in range(100):
                if entry.output_poll_lock.locked():
                    break
                await asyncio.sleep(0.01)
            assert entry.output_poll_lock.locked()

            terminated = await asyncio.wait_for(
                session.pty_terminate(started.process_id),
                timeout=2,
            )
            polled = await asyncio.wait_for(poll_task, timeout=2)

            assert terminated.process_id is None
            assert terminated.exit_code is not None
            assert polled.exit_code is not None
        finally:
            await session.pty_terminate_all()

    def test_pty_pruning_skips_terminating_entries(self, tmp_path: Path) -> None:
        session = _RecordingUnixLocalSession(tmp_path)
        entries: dict[int, _UnixPtyProcessEntry] = {}
        for index in range(PTY_PROCESSES_MAX):
            process = cast(
                asyncio.subprocess.Process,
                SimpleNamespace(returncode=None, pid=None),
            )
            entry = _UnixPtyProcessEntry(process=process, tty=False)
            entry.last_used = time.monotonic() - (PTY_PROCESSES_MAX - index)
            process_id = 1_000 + index
            entries[process_id] = entry

        entries[1_000].termination_pending = True
        session._pty_processes = entries

        pruned = session._prune_pty_processes_if_needed()

        assert pruned is not None
        assert pruned[0] == 1_001
        assert pruned[1] is entries[1_001]
        assert entries[1_001].termination_pending is True
        assert session._pty_processes[1_000] is entries[1_000]

        for entry in entries.values():
            entry.termination_pending = True

        assert session._prune_pty_processes_if_needed() is None

    @pytest.mark.asyncio
    async def test_pty_exec_start_rejects_when_all_entries_are_terminating(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = _RecordingUnixLocalSession(tmp_path)
        process = cast(
            asyncio.subprocess.Process,
            SimpleNamespace(returncode=None, pid=None),
        )
        entry = _UnixPtyProcessEntry(
            process=process,
            tty=False,
            termination_pending=True,
        )
        session._pty_processes[1_000] = entry
        session._reserved_pty_process_ids.add(1_000)
        monkeypatch.setattr(
            "agents.sandbox.sandboxes.unix_local.PTY_PROCESSES_MAX",
            1,
        )

        with pytest.raises(
            RuntimeError,
            match="PTY process limit reached while all registered sessions are terminating",
        ):
            await session.pty_exec_start(
                "sleep",
                "30",
                shell=False,
                tty=False,
                yield_time_s=0.05,
            )

        assert session._pty_processes == {1_000: entry}
        assert session._reserved_pty_process_ids == {1_000}


class TestUnixLocalUserScopedFilesystem:
    @pytest.mark.asyncio
    async def test_mkdir_as_user_checks_permissions_then_uses_local_fs(
        self,
        tmp_path: Path,
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        session = _RecordingUnixLocalSession(workspace)

        await session.mkdir("nested", user=User(name="sandbox-user"))

        assert (workspace / "nested").is_dir()
        assert len(session.exec_commands) == 1
        assert session.exec_commands[0][:4] == ("sudo", "-u", "sandbox-user", "--")
        assert session.exec_commands[0][4:6] == ("sh", "-lc")
        assert session.exec_commands[0][-2:] == (str(workspace / "nested"), "0")
        assert not any(part.startswith("mkdir ") for part in session.exec_commands[0])

    @pytest.mark.asyncio
    async def test_rm_as_user_checks_permissions_then_uses_local_fs(
        self,
        tmp_path: Path,
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "stale.txt"
        target.write_text("stale", encoding="utf-8")
        session = _RecordingUnixLocalSession(workspace)

        await session.rm("stale.txt", user=User(name="sandbox-user"))

        assert not target.exists()
        assert len(session.exec_commands) == 1
        assert session.exec_commands[0][:4] == ("sudo", "-u", "sandbox-user", "--")
        assert session.exec_commands[0][4:6] == ("sh", "-lc")
        assert session.exec_commands[0][-2:] == (str(target), "0")
        assert not any(part.startswith("rm ") for part in session.exec_commands[0])
