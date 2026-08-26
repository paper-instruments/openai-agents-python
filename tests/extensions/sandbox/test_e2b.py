from __future__ import annotations

import asyncio
import base64
import builtins
import inspect
import io
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import pytest
from pydantic import Field, PrivateAttr

import agents.extensions.sandbox.e2b.sandbox as e2b_module
from agents.extensions.sandbox._rclone import (
    ensure_rclone as _ensure_rclone,
    rclone_pattern_for_session as _rclone_pattern_for_session,
)
from agents.extensions.sandbox.e2b.mounts import (
    E2BCloudBucketMountStrategy,
    _assert_e2b_session,
    _ensure_fuse_support,
)
from agents.extensions.sandbox.e2b.sandbox import (
    E2BSandboxClient,
    E2BSandboxClientOptions,
    E2BSandboxSession,
    E2BSandboxSessionState,
)
from agents.sandbox import Manifest
from agents.sandbox.entries import (
    Dir,
    File,
    InContainerMountStrategy,
    Mount,
    MountpointMountPattern,
    RcloneMountPattern,
    S3Mount,
)
from agents.sandbox.entries.mounts.base import InContainerMountAdapter
from agents.sandbox.errors import (
    ExecNonZeroError,
    ExecTimeoutError,
    ExecTransportError,
    InvalidManifestPathError,
    MountConfigError,
    PtySessionNotFoundError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
)
from agents.sandbox.files import EntryKind
from agents.sandbox.materialization import MaterializedFile
from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
from agents.sandbox.session.dependencies import Dependencies
from agents.sandbox.session.pty_types import PtyExecUpdate
from agents.sandbox.session.runtime_helpers import (
    RESOLVE_WORKSPACE_PATH_HELPER,
    WORKSPACE_FINGERPRINT_HELPER,
)
from agents.sandbox.snapshot import NoopSnapshot, SnapshotBase
from agents.sandbox.types import ExecResult, User
from agents.sandbox.workspace_paths import SandboxPathGrant


def test_e2b_package_re_exports_backend_symbols() -> None:
    package_module = __import__(
        "agents.extensions.sandbox.e2b",
        fromlist=["E2BCloudBucketMountStrategy", "E2BSandboxClient"],
    )

    assert package_module.E2BCloudBucketMountStrategy is E2BCloudBucketMountStrategy
    assert package_module.E2BSandboxClient is E2BSandboxClient


def test_e2b_extension_re_exports_cloud_bucket_strategy() -> None:
    package_module = __import__(
        "agents.extensions.sandbox",
        fromlist=["E2BCloudBucketMountStrategy"],
    )

    assert package_module.E2BCloudBucketMountStrategy is E2BCloudBucketMountStrategy


@pytest.mark.skipif(sys.platform != "linux", reason="the E2B supervisor runs on Linux")
def test_e2b_supervisor_retains_ownership_after_direct_child_exits() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            e2b_module._E2B_PROCESS_SUPERVISOR,
            "sh",
            "-c",
            "env -i sleep 30 &",
        ],
        env={e2b_module._E2B_MANAGED_PROCESS_TOKEN_ENV: "test-token"},
        start_new_session=True,
    )
    try:
        time.sleep(0.2)
        assert process.poll() is None
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="the E2B supervisor runs on Linux")
def test_e2b_supervisor_retains_ownership_after_descendant_changes_session() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            e2b_module._E2B_PROCESS_SUPERVISOR,
            "sh",
            "-c",
            "setsid sleep 30 &",
        ],
        env={e2b_module._E2B_MANAGED_PROCESS_TOKEN_ENV: "test-token"},
        start_new_session=True,
    )
    try:
        time.sleep(0.2)
        assert process.poll() is None
    finally:
        subprocess.run(
            [
                sys.executable,
                "-c",
                e2b_module._E2B_PROCESS_GROUP_TERMINATOR,
                "test-token",
                "1",
            ],
            check=True,
            timeout=5,
        )
        process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="the E2B terminator runs on Linux")
def test_e2b_terminator_refreshes_groups_after_term_rehomes_a_process() -> None:
    token = "test-token"
    child = """
import os
import signal
import time


def rehome(_signum, _frame):
    os.setsid()


signal.signal(signal.SIGTERM, rehome)
print("ready", flush=True)
while True:
    time.sleep(1)
"""
    launcher = """
import subprocess
import sys
import time

process = subprocess.Popen(
    [sys.executable, "-c", sys.argv[1]],
    stdout=subprocess.PIPE,
    text=True,
)
assert process.stdout is not None
assert process.stdout.readline() == "ready\\n"
print(process.pid, flush=True)
time.sleep(30)
"""
    environment = {
        **os.environ,
        e2b_module._E2B_MANAGED_PROCESS_TOKEN_ENV: token,
    }
    process = subprocess.Popen(
        [sys.executable, "-c", launcher, child],
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline())
    try:
        subprocess.run(
            [sys.executable, "-c", e2b_module._E2B_PROCESS_GROUP_TERMINATOR, token, "1"],
            check=True,
            timeout=5,
        )
        for _ in range(50):
            try:
                stat = Path(f"/proc/{child_pid}/stat").read_text()
            except FileNotFoundError:
                break
            if stat[stat.rfind(")") + 2 :].split()[0] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("the rehomed managed process survived targeted termination")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_e2b_mount_strategy_type_and_default_pattern() -> None:
    strategy = E2BCloudBucketMountStrategy()

    assert strategy.type == "e2b_cloud_bucket"
    assert isinstance(strategy.pattern, RcloneMountPattern)
    assert strategy.pattern.mode == "fuse"


def test_e2b_mount_strategy_round_trips_through_manifest() -> None:
    manifest = Manifest.model_validate(
        {
            "root": "/workspace",
            "entries": {
                "bucket": {
                    "type": "s3_mount",
                    "bucket": "my-bucket",
                    "mount_strategy": {"type": "e2b_cloud_bucket"},
                }
            },
        }
    )

    mount = manifest.entries["bucket"]
    assert isinstance(mount, S3Mount)
    assert isinstance(mount.mount_strategy, E2BCloudBucketMountStrategy)


def test_e2b_session_guard_rejects_wrong_type() -> None:
    class _WrongSession:
        pass

    with pytest.raises(MountConfigError, match="E2BSandboxSession"):
        _assert_e2b_session(_WrongSession())  # type: ignore[arg-type]


def test_e2b_session_guard_accepts_correct_type() -> None:
    _assert_e2b_session(_FakeMountSession())


@pytest.mark.asyncio
async def test_e2b_ensure_fuse_uses_root_chmod() -> None:
    session = _FakeMountSession([_exec_ok(), _exec_ok()])

    await _ensure_fuse_support(session)

    assert session.exec_calls == [
        (
            "sh -lc test -c /dev/fuse && grep -qw fuse /proc/filesystems && "
            "(command -v fusermount3 >/dev/null 2>&1 || command -v fusermount >/dev/null 2>&1)"
        ),
        (
            "sudo -u root -- sh -lc chmod a+rw /dev/fuse && "
            "touch /etc/fuse.conf && "
            "(grep -qxF user_allow_other /etc/fuse.conf || "
            "printf '\\nuser_allow_other\\n' >> /etc/fuse.conf)"
        ),
    ]


@pytest.mark.asyncio
async def test_e2b_ensure_rclone_installs_with_root_apt() -> None:
    session = _FakeMountSession(
        [
            _exec_fail(),  # rclone missing
            _exec_ok(),  # apt-get present
            _exec_ok(),  # apt-get update succeeds
            _exec_ok(),  # package install succeeds
            _exec_ok(),  # upstream rclone install succeeds
            _exec_ok(),  # rclone now present
        ]
    )

    await _ensure_rclone(session)

    assert session.exec_calls[:2] == [
        "sh -lc command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone",
        "sh -lc command -v apt-get >/dev/null 2>&1",
    ]
    assert session.exec_calls[2] == (
        "sudo -u root -- sh -lc DEBIAN_FRONTEND=noninteractive "
        "DEBCONF_NOWARNINGS=yes apt-get -o Dpkg::Use-Pty=0 update -qq"
    )
    assert session.exec_calls[3] == (
        "sudo -u root -- sh -lc DEBIAN_FRONTEND=noninteractive "
        "DEBCONF_NOWARNINGS=yes apt-get -o Dpkg::Use-Pty=0 install -y -qq "
        "curl unzip ca-certificates"
    )
    assert (
        session.exec_calls[4]
        == "sudo -u root -- sh -lc curl -fsSL https://rclone.org/install.sh | bash"
    )
    assert session.exec_calls[5] == (
        "sh -lc command -v rclone >/dev/null 2>&1 || test -x /usr/local/bin/rclone"
    )


@pytest.mark.asyncio
async def test_e2b_rclone_pattern_adds_fuse_access_args() -> None:
    session = _FakeMountSession([_exec_ok(stdout=b"1000\n1000\n")])

    pattern = await _rclone_pattern_for_session(session, RcloneMountPattern(mode="fuse"))

    assert pattern.extra_args == ["--allow-other", "--uid", "1000", "--gid", "1000"]


@pytest.mark.asyncio
async def test_e2b_rclone_pattern_preserves_explicit_access_args() -> None:
    session = _FakeMountSession([_exec_ok(stdout=b"1000\n1000\n")])
    source_pattern = RcloneMountPattern(
        mode="fuse",
        extra_args=["--allow-other", "--uid", "123", "--gid", "456", "--buffer-size", "0"],
    )

    pattern = await _rclone_pattern_for_session(session, source_pattern)

    assert pattern.extra_args == [
        "--allow-other",
        "--uid",
        "123",
        "--gid",
        "456",
        "--buffer-size",
        "0",
    ]


class _FakeE2BResult:
    def __init__(self, *, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class _FakeE2BCommandExitException(Exception):
    def __init__(self, *, exit_code: int) -> None:
        super().__init__(f"command exited with {exit_code}")
        self.exit_code = exit_code


class _FakeE2BAsyncCommandHandle:
    def __init__(
        self,
        *,
        pid: int = 4242,
        result_exit_code: int = 0,
        initial_exit_code: int | None = None,
        wait_delay_s: float = 0,
        wait_error: BaseException | None = None,
        wait_never: bool = False,
        wait_until_released: bool = False,
    ) -> None:
        self.pid = pid
        self.exit_code = initial_exit_code
        self.result_exit_code = result_exit_code
        self.wait_delay_s = wait_delay_s
        self.wait_error = wait_error
        self.wait_never = wait_never
        self.wait_until_released = wait_until_released
        self.wait_calls = 0
        self.wait_cancelled = False
        self.kill_calls = 0
        self.kill_error: BaseException | None = None
        self.kill_hook: object | None = None
        self._killed = asyncio.Event()
        self._wait_released = asyncio.Event()

    async def wait(self) -> _FakeE2BResult:
        self.wait_calls += 1
        try:
            if self.wait_never:
                await self._killed.wait()
            if self.wait_until_released:
                await self._wait_released.wait()
            if self.wait_delay_s:
                await asyncio.sleep(self.wait_delay_s)
            if self.wait_error is not None:
                raise self.wait_error
            self.exit_code = self.result_exit_code
            return _FakeE2BResult(exit_code=self.result_exit_code)
        except asyncio.CancelledError:
            self.wait_cancelled = True
            raise

    async def kill(self) -> bool:
        self.kill_calls += 1
        if self.kill_error is not None:
            raise self.kill_error
        if callable(self.kill_hook):
            result = self.kill_hook()
            if inspect.isawaitable(result):
                await result
        self._killed.set()
        return True

    def release_wait(self) -> None:
        self._wait_released.set()


class _FakeE2BFiles:
    def __init__(self) -> None:
        self.make_dir_calls: list[tuple[str, float | None]] = []
        self.make_dir_error: BaseException | None = None
        self.write_calls: list[tuple[str, bytes, float | None]] = []
        self.write_files_calls: list[tuple[list[dict[str, object]], float | None]] = []

    async def write(
        self,
        path: str,
        data: bytes,
        request_timeout: float | None = None,
    ) -> None:
        self.write_calls.append((path, data, request_timeout))

    async def write_files(
        self,
        files: Sequence[dict[str, object]],
        request_timeout: float | None = None,
    ) -> None:
        self.write_files_calls.append((list(files), request_timeout))

    async def remove(self, path: str, request_timeout: float | None = None) -> None:
        _ = (path, request_timeout)

    async def make_dir(self, path: str, request_timeout: float | None = None) -> bool:
        self.make_dir_calls.append((path, request_timeout))
        if self.make_dir_error is not None:
            raise self.make_dir_error
        return True

    async def read(self, path: str, format: str = "bytes") -> bytes:
        _ = (path, format)
        return b""


class _FakeE2BCommands:
    def __init__(self) -> None:
        self.exec_root_ready = False
        self.calls: list[dict[str, object]] = []
        self.mkdir_result: _FakeE2BResult | None = None
        self.next_result = _FakeE2BResult()
        self.background_calls: list[dict[str, object]] = []
        self.background_error: BaseException | None = None
        self.background_error_after_start: BaseException | None = None
        self.background_late_error: BaseException | None = None
        self.next_async_command_handle: _FakeE2BAsyncCommandHandle | None = None
        self.async_command_stdout_chunks: list[bytes | str] = []
        self.background_handles: dict[int, _FakeE2BAsyncCommandHandle] = {}
        self.background_tokens: dict[str, _FakeE2BAsyncCommandHandle] = {}
        self.group_termination_calls: list[int] = []
        self.group_termination_users: list[str | None] = []
        self.group_termination_error: BaseException | None = None
        self.group_termination_started = asyncio.Event()
        self.group_termination_release: asyncio.Event | None = None

    async def run(
        self,
        command: str,
        background: bool | None = None,
        envs: dict[str, str] | None = None,
        user: str | None = None,
        cwd: str | None = None,
        on_stdout: object | None = None,
        on_stderr: object | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
        request_timeout: float | None = None,
    ) -> object:
        _ = request_timeout
        if background:
            if self.background_error is not None:
                raise self.background_error
            _ = on_stderr
            self.background_calls.append(
                {
                    "command": command,
                    "timeout": timeout,
                    "cwd": cwd,
                    "envs": envs,
                    "stdin": stdin,
                    "background": background,
                }
            )
            if callable(on_stdout):
                for chunk in self.async_command_stdout_chunks:
                    result = on_stdout(chunk)
                    if inspect.isawaitable(result):
                        await result

            handle = self.next_async_command_handle or _FakeE2BAsyncCommandHandle()
            self.background_handles[handle.pid] = handle
            process_token = (envs or {}).get(e2b_module._E2B_MANAGED_PROCESS_TOKEN_ENV)
            if self.background_late_error is not None:
                assert process_token is not None

                async def publish_late_process() -> None:
                    await self.group_termination_started.wait()
                    await asyncio.sleep(0)
                    self.background_tokens[process_token] = handle

                asyncio.create_task(publish_late_process())
                raise self.background_late_error
            if process_token is not None:
                self.background_tokens[process_token] = handle
            if self.background_error_after_start is not None:
                raise self.background_error_after_start
            return handle

        self.calls.append(
            {
                "command": command,
                "timeout": timeout,
                "cwd": cwd,
                "envs": envs,
                "user": user,
            }
        )
        parts = shlex.split(command)
        if (
            len(parts) == 5
            and parts[:2] == ["python3", "-c"]
            and parts[2] == e2b_module._E2B_PROCESS_GROUP_TERMINATOR
        ):
            self.group_termination_started.set()
            background_handle = None
            for _ in range(int(parts[4])):
                background_handle = self.background_tokens.get(parts[3])
                if background_handle is not None:
                    break
                await asyncio.sleep(0)
            if background_handle is not None:
                self.group_termination_calls.append(background_handle.pid)
            self.group_termination_users.append(user)
            if self.group_termination_error is not None:
                raise self.group_termination_error
            if background_handle is not None:
                background_handle._killed.set()
                background_handle.release_wait()
            if self.group_termination_release is not None:
                await self.group_termination_release.wait()
            return _FakeE2BResult()
        if _is_helper_install_command(command):
            return _FakeE2BResult()
        if _is_helper_present_command(command):
            return _FakeE2BResult()
        if parts and parts[0] == str(RESOLVE_WORKSPACE_PATH_HELPER.install_path):
            return _FakeE2BResult(stdout=parts[2])
        if parts and parts[0] == str(WORKSPACE_FINGERPRINT_HELPER.install_path):
            return _FakeE2BResult(
                stdout='{"fingerprint":"fake-workspace-fingerprint","version":"workspace_tar_sha256_v1"}\n'
            )
        if command == "test -d /workspace" and cwd in (None, "/"):
            exit_code = 0 if self.exec_root_ready else 1
            return _FakeE2BResult(exit_code=exit_code)
        if command == "mkdir -p -- /workspace" and cwd == "/":
            result = self.mkdir_result or _FakeE2BResult()
            if result.exit_code == 0:
                self.exec_root_ready = True
            self.mkdir_result = None
            return result
        if cwd == "/workspace" and not self.exec_root_ready:
            raise ValueError("cwd '/workspace' does not exist")
        result = self.next_result
        self.next_result = _FakeE2BResult()
        return result


class _FakeE2BPtyHandle(_FakeE2BAsyncCommandHandle):
    def __init__(
        self,
        *,
        result_exit_code: int = 0,
        wait_delay_s: float = 0,
        wait_error: BaseException | None = None,
        wait_never: bool = True,
    ) -> None:
        super().__init__(
            result_exit_code=result_exit_code,
            wait_delay_s=wait_delay_s,
            wait_error=wait_error,
            wait_never=wait_never,
        )
        self.pid = "pty-123"  # type: ignore[assignment]
        self.stdin_payloads: list[bytes] = []


class _FakeE2BPty:
    def __init__(self) -> None:
        self.handle = _FakeE2BPtyHandle()
        self.commands: _FakeE2BCommands | None = None
        self.on_data: object | None = None
        self.stdin_output_chunks: list[bytes | str] = []
        self.create_error: BaseException | None = None
        self.create_late_error: BaseException | None = None
        self.send_stdin_error: BaseException | None = None

    async def create(
        self,
        *,
        size: object,
        cwd: str | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
        on_data: object | None = None,
    ) -> _FakeE2BPtyHandle:
        _ = (size, cwd, envs, timeout)
        if self.create_error is not None:
            raise self.create_error
        process_token = (envs or {}).get(e2b_module._E2B_MANAGED_PROCESS_TOKEN_ENV)
        if self.create_late_error is not None:
            assert process_token is not None
            assert self.commands is not None
            commands = self.commands

            async def publish_late_process() -> None:
                await commands.group_termination_started.wait()
                await asyncio.sleep(0)
                commands.background_tokens[process_token] = self.handle

            asyncio.create_task(publish_late_process())
            raise self.create_late_error
        if process_token is not None:
            assert self.commands is not None
            self.commands.background_tokens[process_token] = self.handle
        self.on_data = on_data
        return self.handle

    async def send_stdin(
        self,
        pid: object,
        data: bytes,
        request_timeout: float | None = None,
    ) -> None:
        _ = (pid, request_timeout)
        if self.send_stdin_error is not None:
            raise self.send_stdin_error
        self.handle.stdin_payloads.append(data)
        if callable(self.on_data):
            for chunk in self.stdin_output_chunks:
                result = self.on_data(chunk)
                if inspect.isawaitable(result):
                    await result
            self.stdin_output_chunks.clear()


class _FakeE2BSandbox:
    def __init__(self) -> None:
        self.sandbox_id = "sb-123"
        self.files = _FakeE2BFiles()
        self.commands = _FakeE2BCommands()
        self.pty = _FakeE2BPty()
        self.pty.commands = self.commands
        self.created_snapshot_id = "snap-123"
        self.pause_error: BaseException | None = None
        self.kill_error: BaseException | None = None
        self.pause_calls = 0
        self.kill_calls = 0

    async def pause(self) -> None:
        self.pause_calls += 1
        if self.pause_error is not None:
            raise self.pause_error
        return

    async def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_error is not None:
            raise self.kill_error
        return

    async def is_running(self, request_timeout: float | None = None) -> bool:
        _ = request_timeout
        return True

    def get_host(self, port: int) -> str:
        return f"{port}-{self.sandbox_id}.sandbox.example.test"

    async def create_snapshot(self) -> object:
        return type("SnapshotInfo", (), {"snapshot_id": self.created_snapshot_id})()


class _FakeMountSession(BaseSandboxSession):
    __name__ = "E2BSandboxSession"

    def __init__(self, results: list[ExecResult] | None = None) -> None:
        self.state = E2BSandboxSessionState(
            session_id=uuid.uuid4(),
            manifest=Manifest(root="/workspace"),
            snapshot=NoopSnapshot(id="snapshot"),
            sandbox_id="sb-123",
        )
        self._results = list(results or [])
        self.exec_calls: list[str] = []

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        _ = timeout
        cmd_str = " ".join(str(c) for c in command)
        self.exec_calls.append(cmd_str)
        if self._results:
            return self._results.pop(0)
        return ExecResult(stdout=b"", stderr=b"", exit_code=0)

    async def read(self, path: Path, *, user: str | User | None = None) -> io.IOBase:
        _ = (path, user)
        return io.BytesIO(b"")

    async def write(self, path: Path, data: io.IOBase, *, user: str | User | None = None) -> None:
        _ = (path, data, user)

    async def persist_workspace(self) -> io.IOBase:
        raise AssertionError("not expected")

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        _ = data
        raise AssertionError("not expected")

    async def running(self) -> bool:
        return True


_FakeMountSession.__name__ = "E2BSandboxSession"


def _exec_ok(stdout: bytes = b"") -> ExecResult:
    return ExecResult(stdout=stdout, stderr=b"", exit_code=0)


def _exec_fail() -> ExecResult:
    return ExecResult(stdout=b"", stderr=b"", exit_code=1)


class _RestorableSnapshot(SnapshotBase):
    type: Literal["test-restorable-e2b"] = "test-restorable-e2b"
    payload: bytes = b"restored"

    async def persist(self, data: io.IOBase, *, dependencies: Dependencies | None = None) -> None:
        _ = (data, dependencies)

    async def restore(self, *, dependencies: Dependencies | None = None) -> io.IOBase:
        _ = dependencies
        return io.BytesIO(self.payload)

    async def restorable(self, *, dependencies: Dependencies | None = None) -> bool:
        _ = dependencies
        return True


class _RecordingMount(Mount):
    type: str = "recording_mount"
    mount_strategy: InContainerMountStrategy = Field(
        default_factory=lambda: InContainerMountStrategy(pattern=MountpointMountPattern())
    )
    _mounted_paths: list[Path] = PrivateAttr(default_factory=list)
    _unmounted_paths: list[Path] = PrivateAttr(default_factory=list)
    _events: list[tuple[str, str]] = PrivateAttr(default_factory=list)

    def bind_events(self, events: list[tuple[str, str]]) -> _RecordingMount:
        self._events = events
        return self

    def supported_in_container_patterns(
        self,
    ) -> tuple[builtins.type[MountpointMountPattern], ...]:
        return (MountpointMountPattern,)

    def build_docker_volume_driver_config(
        self,
        strategy: object,
    ) -> tuple[str, dict[str, str], bool]:
        _ = strategy
        raise MountConfigError(
            message="docker-volume mounts are not supported for this mount type",
            context={"mount_type": self.type},
        )

    def in_container_adapter(self) -> InContainerMountAdapter:
        mount = self

        class _Adapter(InContainerMountAdapter):
            def validate(self, strategy: InContainerMountStrategy) -> None:
                _ = strategy

            async def activate(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                dest: Path,
                base_dir: Path,
            ) -> list[MaterializedFile]:
                _ = (strategy, session, base_dir)
                path = mount._resolve_mount_path(session, dest)
                mount._events.append(("mount", path.as_posix()))
                mount._mounted_paths.append(path)
                return []

            async def deactivate(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                dest: Path,
                base_dir: Path,
            ) -> None:
                _ = (strategy, session, base_dir)
                path = mount._resolve_mount_path(session, dest)
                mount._events.append(("unmount", path.as_posix()))
                mount._unmounted_paths.append(path)

            async def teardown_for_snapshot(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                path: Path,
            ) -> None:
                _ = (strategy, session)
                mount._events.append(("unmount", path.as_posix()))
                mount._unmounted_paths.append(path)

            async def restore_after_snapshot(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                path: Path,
            ) -> None:
                _ = (strategy, session)
                mount._events.append(("mount", path.as_posix()))
                mount._mounted_paths.append(path)

        return _Adapter(self)


class _FailingUnmountMount(_RecordingMount):
    type: str = "failing_unmount_mount"

    def in_container_adapter(self) -> InContainerMountAdapter:
        mount = self
        base_adapter = super().in_container_adapter()

        class _Adapter(InContainerMountAdapter):
            def validate(self, strategy: InContainerMountStrategy) -> None:
                base_adapter.validate(strategy)

            async def activate(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                dest: Path,
                base_dir: Path,
            ) -> list[MaterializedFile]:
                return await base_adapter.activate(strategy, session, dest, base_dir)

            async def deactivate(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                dest: Path,
                base_dir: Path,
            ) -> None:
                _ = (strategy, session, base_dir)
                path = mount._resolve_mount_path(session, dest)
                mount._events.append(("unmount_fail", path.as_posix()))
                raise RuntimeError("boom while unmounting second mount")

            async def teardown_for_snapshot(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                path: Path,
            ) -> None:
                _ = (strategy, session)
                mount._events.append(("unmount_fail", path.as_posix()))
                raise RuntimeError("boom while unmounting second mount")

            async def restore_after_snapshot(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                path: Path,
            ) -> None:
                await base_adapter.restore_after_snapshot(strategy, session, path)

        return _Adapter(self)


class _FailingRemountMount(_RecordingMount):
    type: str = "failing_remount_mount"

    def in_container_adapter(self) -> InContainerMountAdapter:
        mount = self
        base_adapter = super().in_container_adapter()

        class _Adapter(InContainerMountAdapter):
            def validate(self, strategy: InContainerMountStrategy) -> None:
                base_adapter.validate(strategy)

            async def activate(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                dest: Path,
                base_dir: Path,
            ) -> list[MaterializedFile]:
                _ = (strategy, session, base_dir)
                path = mount._resolve_mount_path(session, dest)
                mount._events.append(("mount_fail", path.as_posix()))
                raise RuntimeError("boom while remounting second mount")

            async def deactivate(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                dest: Path,
                base_dir: Path,
            ) -> None:
                return await base_adapter.deactivate(strategy, session, dest, base_dir)

            async def teardown_for_snapshot(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                path: Path,
            ) -> None:
                await base_adapter.teardown_for_snapshot(strategy, session, path)

            async def restore_after_snapshot(
                self,
                strategy: InContainerMountStrategy,
                session: BaseSandboxSession,
                path: Path,
            ) -> None:
                _ = (strategy, session)
                mount._events.append(("mount_fail", path.as_posix()))
                raise RuntimeError("boom while remounting second mount")

        return _Adapter(self)


def _session(
    *,
    workspace_root_ready: bool = False,
    exposed_ports: tuple[int, ...] = (),
) -> tuple[E2BSandboxSession, _FakeE2BSandbox]:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=workspace_root_ready,
        exposed_ports=exposed_ports,
    )
    return E2BSandboxSession.from_state(state, sandbox=sandbox), sandbox


class _LocalStatE2BSession(E2BSandboxSession):
    def __init__(self, *, state: E2BSandboxSessionState, sandbox: object) -> None:
        super().__init__(state=state, sandbox=sandbox)
        self.stat_exec_calls: list[tuple[str, ...]] = []

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        rendered = tuple(str(part) for part in command)
        self.stat_exec_calls.append(rendered)
        process = await asyncio.create_subprocess_exec(
            *rendered,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=process.returncode or 0)


def _local_stat_session(
    workspace: Path,
    *,
    extra_path_grants: tuple[SandboxPathGrant, ...] = (),
) -> _LocalStatE2BSession:
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root=str(workspace), extra_path_grants=extra_path_grants),
        snapshot=NoopSnapshot(id="stat-snapshot"),
        sandbox_id="sb-local-stat",
        workspace_root_ready=True,
    )
    return _LocalStatE2BSession(state=state, sandbox=_FakeE2BSandbox())


def _stat_entry_stdout(*, mode: int = 0o100640, size: int = 4) -> str:
    return json.dumps(
        {
            "group": "runner",
            "mode": mode,
            "owner": "runner",
            "size": size,
            "status": "entry",
        }
    )


@pytest.mark.skipif(sys.platform == "win32", reason="the E2B worker runs on Linux")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry_kind", "expected_size"),
    [
        (EntryKind.FILE, 5),
        (EntryKind.DIRECTORY, None),
        (EntryKind.OTHER, None),
    ],
)
async def test_e2b_stat_returns_metadata_with_one_routed_exec(
    tmp_path: Path,
    entry_kind: EntryKind,
    expected_size: int | None,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target"
    if entry_kind is EntryKind.FILE:
        target.write_bytes(b"hello")
    elif entry_kind is EntryKind.DIRECTORY:
        target.mkdir()
    else:
        os.mkfifo(target)
    session = _local_stat_session(workspace)

    result = await session.stat("target")

    assert result is not None
    assert result.path == target.as_posix()
    assert result.kind is entry_kind
    assert result.permissions.directory is (entry_kind is EntryKind.DIRECTORY)
    if expected_size is not None:
        assert result.size == expected_size
    assert len(session.stat_exec_calls) == 1
    assert session.stat_exec_calls[0][:2] == ("python3", "-c")


@pytest.mark.skipif(sys.platform == "win32", reason="the E2B worker runs on Linux")
@pytest.mark.asyncio
async def test_e2b_stat_returns_none_only_for_missing_final_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = _local_stat_session(workspace)

    result = await session.stat("missing.txt")

    assert result is None
    assert len(session.stat_exec_calls) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="the E2B worker runs on Linux")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parent_kind", "reason"),
    [
        ("missing", "missing_ancestor"),
        ("dangling_symlink", "missing_ancestor"),
        ("file", "not_directory"),
    ],
)
async def test_e2b_stat_distinguishes_invalid_ancestors(
    tmp_path: Path,
    parent_kind: str,
    reason: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if parent_kind == "file":
        (workspace / "parent").write_text("not a directory", encoding="utf-8")
    elif parent_kind == "dangling_symlink":
        (workspace / "parent").symlink_to("missing", target_is_directory=True)
    session = _local_stat_session(workspace)

    with pytest.raises(WorkspaceReadNotFoundError) as exc_info:
        await session.stat("parent/target.txt")

    assert exc_info.value.context["reason"] == reason
    assert exc_info.value.context["component"] == (workspace / "parent").as_posix()
    assert len(session.stat_exec_calls) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="the E2B worker runs on Linux")
@pytest.mark.asyncio
async def test_e2b_stat_follows_safe_parent_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real_parent = workspace / "real"
    real_parent.mkdir()
    (real_parent / "target.txt").write_text("content", encoding="utf-8")
    (workspace / "parent").symlink_to(real_parent, target_is_directory=True)
    session = _local_stat_session(workspace)

    result = await session.stat("parent/target.txt")

    assert result is not None
    assert result.path == (workspace / "parent" / "target.txt").as_posix()
    assert result.kind is EntryKind.FILE
    assert result.size == len("content")
    assert len(session.stat_exec_calls) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="the E2B worker runs on Linux")
@pytest.mark.asyncio
@pytest.mark.parametrize("link_as_parent", [False, True])
async def test_e2b_stat_rejects_remote_symlink_escape_with_one_exec(
    tmp_path: Path,
    link_as_parent: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.txt").write_text("secret", encoding="utf-8")
    if link_as_parent:
        link = workspace / "link"
        link.symlink_to(outside, target_is_directory=True)
        requested_path = "link/target.txt"
    else:
        link = workspace / "link.txt"
        link.symlink_to(outside / "target.txt")
        requested_path = "link.txt"
    session = _local_stat_session(workspace)

    with pytest.raises(InvalidManifestPathError) as exc_info:
        await session.stat(requested_path)

    assert exc_info.value.context["resolved_path"] == (outside / "target.txt").as_posix()
    assert len(session.stat_exec_calls) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="the E2B worker runs on Linux")
@pytest.mark.asyncio
async def test_e2b_stat_follows_safe_leaf_symlink_and_reports_dangling_target_missing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "target.txt").write_text("content", encoding="utf-8")
    link = workspace / "link.txt"
    link.symlink_to("target.txt")
    session = _local_stat_session(workspace)

    result = await session.stat("link.txt")

    assert result is not None
    assert result.path == link.as_posix()
    assert result.kind is EntryKind.FILE
    assert result.size == len("content")
    assert len(session.stat_exec_calls) == 1

    dangling = workspace / "dangling.txt"
    dangling.symlink_to("missing.txt")
    dangling_result = await session.stat("dangling.txt")
    assert dangling_result is None
    assert len(session.stat_exec_calls) == 2


@pytest.mark.skipif(sys.platform == "win32", reason="the E2B worker runs on Linux")
@pytest.mark.asyncio
async def test_e2b_stat_honors_grants_and_rejects_lexical_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    grant = tmp_path / "grant"
    grant.mkdir()
    granted_file = grant / "allowed.txt"
    granted_file.write_text("allowed", encoding="utf-8")
    session = _local_stat_session(
        workspace,
        extra_path_grants=(SandboxPathGrant(path=grant.as_posix()),),
    )

    result = await session.stat(granted_file)

    assert result is not None
    assert result.path == granted_file.as_posix()
    assert result.kind is EntryKind.FILE
    assert len(session.stat_exec_calls) == 1
    with pytest.raises(InvalidManifestPathError):
        await session.stat("../outside.txt")
    assert len(session.stat_exec_calls) == 1

    root_grant = tmp_path / "root-grant"
    root_grant.symlink_to("/", target_is_directory=True)
    invalid_grant_session = _local_stat_session(
        workspace,
        extra_path_grants=(SandboxPathGrant(path=root_grant.as_posix()),),
    )
    with pytest.raises(ValueError, match="must not resolve to filesystem root"):
        await invalid_grant_session.stat("allowed.txt")
    assert len(invalid_grant_session.stat_exec_calls) == 1

    real_grant = tmp_path / "real-grant"
    real_grant.mkdir()
    (real_grant / "child.txt").write_text("child", encoding="utf-8")
    symlink_grant = tmp_path / "symlink-grant"
    symlink_grant.symlink_to(real_grant, target_is_directory=True)
    symlink_grant_session = _local_stat_session(
        workspace,
        extra_path_grants=(SandboxPathGrant(path=symlink_grant.as_posix()),),
    )
    symlink_grant_result = await symlink_grant_session.stat(symlink_grant / "child.txt")
    assert symlink_grant_result is not None
    assert symlink_grant_result.kind is EntryKind.FILE
    assert len(symlink_grant_session.stat_exec_calls) == 1

    exact_grant_session = _local_stat_session(
        workspace,
        extra_path_grants=(SandboxPathGrant(path=symlink_grant.as_posix()),),
    )
    exact_grant_result = await exact_grant_session.stat(symlink_grant)
    assert exact_grant_result is not None
    assert exact_grant_result.kind is EntryKind.DIRECTORY
    assert len(exact_grant_session.stat_exec_calls) == 1

    dangling_grant = tmp_path / "dangling-grant"
    dangling_grant.symlink_to(tmp_path / "missing-grant", target_is_directory=True)
    dangling_grant_session = _local_stat_session(
        workspace,
        extra_path_grants=(SandboxPathGrant(path=dangling_grant.as_posix()),),
    )
    with pytest.raises(WorkspaceReadNotFoundError) as exc_info:
        await dangling_grant_session.stat(dangling_grant / "child.txt")
    assert exc_info.value.context["reason"] == "missing_ancestor"
    assert len(dangling_grant_session.stat_exec_calls) == 1

    blocking_parent = tmp_path / "blocking-parent"
    blocking_parent.write_text("not a directory", encoding="utf-8")
    blocked_grant = blocking_parent / "grant"
    blocked_grant_session = _local_stat_session(
        workspace,
        extra_path_grants=(SandboxPathGrant(path=blocked_grant.as_posix()),),
    )
    with pytest.raises(WorkspaceReadNotFoundError) as exc_info:
        await blocked_grant_session.stat(blocked_grant / "child.txt")
    assert exc_info.value.context["reason"] == "not_directory"
    assert len(blocked_grant_session.stat_exec_calls) == 1


@pytest.mark.asyncio
async def test_e2b_stat_passes_optional_user_through_provider_exec() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(stdout=_stat_entry_stdout(mode=0o140770))

    result = await session.stat("report.txt", user=User(name="runner"))

    assert result is not None
    assert result.path == "/workspace/report.txt"
    assert result.kind is EntryKind.OTHER
    assert result.permissions.directory is False
    assert len(sandbox.commands.calls) == 1
    call = sandbox.commands.calls[0]
    assert shlex.split(str(call["command"]))[:2] == ["python3", "-c"]
    assert call["user"] == "runner"


@pytest.mark.asyncio
async def test_e2b_stat_propagates_operational_failure() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(stderr="permission denied", exit_code=13)

    with pytest.raises(ExecNonZeroError, match="permission denied"):
        await session.stat("report.txt")

    assert len(sandbox.commands.calls) == 1


@pytest.mark.asyncio
async def test_e2b_stat_rejects_malformed_provider_output() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(stdout="not-json")

    with pytest.raises(ExecTransportError) as exc_info:
        await session.stat("report.txt")

    assert exc_info.value.context["reason"] == "malformed_stat_response"
    assert len(sandbox.commands.calls) == 1


def _tar_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("note.txt")
        payload = b"hello"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_e2b_manifest_uses_native_bulk_write_for_files() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True
    session.state.manifest = Manifest(
        root="/workspace",
        entries={
            "skills/alpha.md": File(content=b"alpha"),
            "skills/beta.md": File(content=b"beta"),
        },
    )

    result = await session.apply_manifest()

    assert result.files == []
    assert sandbox.files.write_calls == []
    assert sandbox.files.write_files_calls == [
        (
            [
                {"path": "/workspace/skills/alpha.md", "data": b"alpha"},
                {"path": "/workspace/skills/beta.md", "data": b"beta"},
            ],
            session.state.timeouts.file_upload_s,
        )
    ]


@pytest.mark.asyncio
async def test_e2b_manifest_bulk_write_skips_remote_path_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, sandbox = _session(workspace_root_ready=True)

    async def fail_validation(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("manifest bulk writes must not perform remote path validation")

    monkeypatch.setattr(session, "_validate_path_access", fail_validation)

    await session._write_file_batch_immediately(
        [
            (Path("/workspace/.agents/alpha/SKILL.md"), b"alpha"),
            (Path("/workspace/.agents/beta/SKILL.md"), b"beta"),
        ]
    )

    assert sandbox.files.write_files_calls == [
        (
            [
                {"path": "/workspace/.agents/alpha/SKILL.md", "data": b"alpha"},
                {"path": "/workspace/.agents/beta/SKILL.md", "data": b"beta"},
            ],
            session.state.timeouts.file_upload_s,
        )
    ]


@pytest.mark.asyncio
async def test_e2b_manifest_uses_one_bulk_write_across_nested_directories() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True
    session.state.manifest = Manifest(
        root="/workspace",
        entries={
            ".agents/alpha": Dir(
                children={
                    "SKILL.md": File(content=b"alpha"),
                    "references": Dir(
                        children={
                            "one.md": File(content=b"one"),
                            "two.md": File(content=b"two"),
                        }
                    ),
                }
            ),
            ".agents/beta": Dir(
                children={"SKILL.md": File(content=b"beta")},
            ),
        },
    )

    result = await session.apply_manifest()

    assert result.files == []
    assert sandbox.files.write_calls == []
    assert len(sandbox.files.write_files_calls) == 1
    uploaded_files, request_timeout = sandbox.files.write_files_calls[0]
    assert request_timeout == session.state.timeouts.file_upload_s
    assert {str(file["path"]): file["data"] for file in uploaded_files} == {
        "/workspace/.agents/alpha/SKILL.md": b"alpha",
        "/workspace/.agents/alpha/references/one.md": b"one",
        "/workspace/.agents/alpha/references/two.md": b"two",
        "/workspace/.agents/beta/SKILL.md": b"beta",
    }
    chmod_commands = [
        call["command"]
        for call in sandbox.commands.calls
        if str(call["command"]).startswith("chmod ")
    ]
    assert chmod_commands == ["chmod -R 0755 /workspace/.agents"]


@pytest.mark.asyncio
async def test_e2b_sandbox_connect_prefers_full_sandbox_wrapper() -> None:
    class _FakeSandboxClass:
        calls: list[tuple[str, str, int | None]] = []

        @classmethod
        async def connect(cls, *, sandbox_id: str, timeout: int | None = None) -> str:
            cls.calls.append(("connect", sandbox_id, timeout))
            return "full-sandbox-wrapper"

        @classmethod
        async def _cls_connect_sandbox(cls, *, sandbox_id: str, timeout: int | None = None) -> str:
            cls.calls.append(("_cls_connect_sandbox", sandbox_id, timeout))
            return "private-full-sandbox-wrapper"

        @classmethod
        async def _cls_connect(cls, *, sandbox_id: str, timeout: int | None = None) -> str:
            cls.calls.append(("_cls_connect", sandbox_id, timeout))
            return "low-level-api-model"

    connected = await e2b_module._sandbox_connect(
        cast(e2b_module._E2BSandboxFactoryAPI, _FakeSandboxClass),
        sandbox_id="sb-123",
        timeout=300,
    )

    assert connected == "full-sandbox-wrapper"
    assert _FakeSandboxClass.calls == [("connect", "sb-123", 300)]


def test_e2b_import_resolves_sdk_sandbox_classes_for_canonical_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "e2b_code_interpreter":
            imports.append(name)
            return type("FakeCodeInterpreterModule", (), {"AsyncSandbox": object()})()
        if name == "e2b":
            imports.append(name)
            return type("FakeE2BModule", (), {"AsyncSandbox": object()})()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    assert e2b_module._import_sandbox_class(e2b_module.E2BSandboxType.CODE_INTERPRETER) is not None
    assert e2b_module._import_sandbox_class(e2b_module.E2BSandboxType.E2B) is not None
    assert imports == ["e2b_code_interpreter", "e2b"]


def _visible_command_calls(sandbox: _FakeE2BSandbox) -> list[dict[str, object]]:
    return [
        call
        for call in sandbox.commands.calls
        if not _is_helper_install_command(str(call["command"]))
        and not _is_helper_present_command(str(call["command"]))
        and not _is_helper_invoke_command(str(call["command"]))
    ]


def _is_helper_install_command(command: str) -> bool:
    return RESOLVE_WORKSPACE_PATH_HELPER.install_marker in command


def _is_helper_invoke_command(command: str) -> bool:
    parts = shlex.split(command)
    return bool(parts) and parts[0].startswith("/tmp/openai-agents/bin/")


def _is_helper_present_command(command: str) -> bool:
    parts = shlex.split(command)
    return (
        len(parts) == 3
        and parts[:2] == ["test", "-x"]
        and parts[2].startswith("/tmp/openai-agents/bin/")
    )


@pytest.mark.asyncio
async def test_e2b_exec_omits_cwd_until_workspace_ready() -> None:
    session, sandbox = _session(workspace_root_ready=False)

    result = await session._exec_internal("find", ".", timeout=0.01)  # noqa: SLF001

    assert result.ok()
    assert sandbox.commands.calls == [
        {
            "command": "find .",
            "timeout": 0.01,
            "cwd": None,
            "envs": {},
            "user": None,
        }
    ]


@pytest.mark.asyncio
async def test_e2b_exec_uses_manifest_root_after_workspace_ready() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True

    result = await session._exec_internal("find", ".", timeout=0.01)  # noqa: SLF001

    assert result.ok()
    assert sandbox.commands.calls == [
        {
            "command": "find .",
            "timeout": 0.01,
            "cwd": "/workspace",
            "envs": {},
            "user": None,
        }
    ]


@pytest.mark.asyncio
async def test_e2b_start_prepares_workspace_root_for_command_cwd() -> None:
    session, sandbox = _session(workspace_root_ready=False)

    await session.start()
    result = await session._exec_internal("pwd", timeout=0.01)  # noqa: SLF001

    assert result.ok()
    assert sandbox.files.make_dir_calls == [("/workspace", 10)]
    assert session.state.workspace_root_ready is True
    assert session._workspace_root_ready is True  # noqa: SLF001
    assert _visible_command_calls(sandbox) == [
        {
            "command": "test -d /workspace",
            "timeout": 10.0,
            "cwd": None,
            "envs": {},
            "user": None,
        },
        {
            "command": "mkdir -p -- /workspace",
            "timeout": 10,
            "cwd": "/",
            "envs": {},
            "user": None,
        },
        {
            "command": "pwd",
            "timeout": 0.01,
            "cwd": "/workspace",
            "envs": {},
            "user": None,
        },
    ]


@pytest.mark.asyncio
async def test_e2b_start_skips_files_api_when_workspace_root_exists() -> None:
    session, sandbox = _session(workspace_root_ready=False)
    sandbox.commands.exec_root_ready = True
    sandbox.files.make_dir_error = TimeoutError("files API unavailable")

    await session.start()

    assert sandbox.files.make_dir_calls == []
    assert session.state.workspace_root_ready is True
    assert session._workspace_root_ready is True  # noqa: SLF001
    assert _visible_command_calls(sandbox)[:2] == [
        {
            "command": "test -d /workspace",
            "timeout": 10.0,
            "cwd": None,
            "envs": {},
            "user": None,
        },
        {
            "command": "mkdir -p -- /workspace",
            "timeout": 10,
            "cwd": "/",
            "envs": {},
            "user": None,
        },
    ]


@pytest.mark.asyncio
async def test_e2b_mkdir_recreates_workspace_root_when_readiness_is_stale() -> None:
    session, sandbox = _session(workspace_root_ready=False)
    sandbox.commands.exec_root_ready = True

    await session.start()
    sandbox.commands.exec_root_ready = False
    command_calls_before_recovery = list(sandbox.commands.calls)
    await session.mkdir("/workspace", parents=True)

    assert sandbox.files.make_dir_calls == [("/workspace", 10)]
    assert sandbox.commands.calls == command_calls_before_recovery


@pytest.mark.asyncio
async def test_e2b_start_installs_runtime_helpers() -> None:
    session, sandbox = _session(workspace_root_ready=False)

    await session.start()

    assert any(_is_helper_install_command(str(call["command"])) for call in sandbox.commands.calls)


@pytest.mark.asyncio
async def test_e2b_start_raises_on_nonzero_workspace_root_setup_exit() -> None:
    session, sandbox = _session(workspace_root_ready=False)
    sandbox.commands.mkdir_result = _FakeE2BResult(stderr="mkdir failed", exit_code=2)

    with pytest.raises(WorkspaceStartError) as exc_info:
        await session.start()

    assert exc_info.value.context["reason"] == "workspace_root_nonzero_exit"
    assert exc_info.value.context["exit_code"] == 2
    assert session.state.workspace_root_ready is False
    assert session._workspace_root_ready is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_preserved_start_still_prepares_workspace_root_for_resumed_exec_cwd() -> None:
    session, sandbox = _session(workspace_root_ready=False)
    session._set_start_state_preserved(True)  # noqa: SLF001

    await session.start()
    result = await session._exec_internal("pwd", timeout=0.01)  # noqa: SLF001

    assert result.ok()
    assert session.state.workspace_root_ready is True
    assert session._workspace_root_ready is True  # noqa: SLF001
    assert session._can_reuse_preserved_workspace_on_resume() is False  # noqa: SLF001
    assert session.should_provision_manifest_accounts_on_resume() is False
    assert _visible_command_calls(sandbox) == [
        {
            "command": "test -d /workspace",
            "timeout": 10.0,
            "cwd": None,
            "envs": {},
            "user": None,
        },
        {
            "command": "mkdir -p -- /workspace",
            "timeout": 10,
            "cwd": "/",
            "envs": {},
            "user": None,
        },
        {
            "command": "pwd",
            "timeout": 0.01,
            "cwd": "/workspace",
            "envs": {},
            "user": None,
        },
    ]


@pytest.mark.asyncio
async def test_e2b_preserved_start_uses_shared_resume_gate_for_restore() -> None:
    session, _sandbox = _session(workspace_root_ready=True)
    session.state.snapshot = _RestorableSnapshot(id="snapshot")
    session._set_start_state_preserved(True)  # noqa: SLF001
    events: list[object] = []

    async def _gate(*, is_running: bool) -> bool:
        events.append(("gate", is_running))
        return False

    async def _restore() -> None:
        events.append("restore")

    async def _reapply() -> None:
        events.append("reapply")

    session._can_skip_snapshot_restore_on_resume = _gate  # type: ignore[method-assign]
    session._restore_snapshot_into_workspace_on_resume = _restore  # type: ignore[method-assign]
    session._reapply_ephemeral_manifest_on_resume = _reapply  # type: ignore[method-assign]

    await session.start()

    assert session.state.workspace_root_ready is True
    assert session._workspace_root_ready is True  # noqa: SLF001
    assert events == [("gate", True), "restore", "reapply"]


@pytest.mark.asyncio
async def test_e2b_running_requires_workspace_root_ready() -> None:
    session, _sandbox = _session(workspace_root_ready=False)

    assert await session.running() is False


@pytest.mark.asyncio
async def test_e2b_running_checks_remote_after_workspace_ready() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True

    assert await session.running() is True


@pytest.mark.asyncio
async def test_e2b_resolve_exposed_port_uses_backend_host() -> None:
    session, _sandbox = _session(workspace_root_ready=True, exposed_ports=(8765,))

    endpoint = await session.resolve_exposed_port(8765)

    assert endpoint.host == "8765-sb-123.sandbox.example.test"
    assert endpoint.port == 443
    assert endpoint.tls is True


@pytest.mark.asyncio
async def test_e2b_client_create_enables_public_traffic_for_exposed_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_calls: list[dict[str, object]] = []

    class _FakeSandboxFactory:
        @staticmethod
        async def create(
            *,
            template: str | None = None,
            timeout: int | None = None,
            metadata: dict[str, str] | None = None,
            envs: dict[str, str] | None = None,
            secure: bool = True,
            allow_internet_access: bool = True,
            network: dict[str, object] | None = None,
            lifecycle: dict[str, object] | None = None,
            mcp: dict[str, dict[str, str]] | None = None,
        ) -> _FakeE2BSandbox:
            _ = (
                template,
                timeout,
                metadata,
                envs,
                secure,
                allow_internet_access,
                network,
                lifecycle,
                mcp,
            )
            create_calls.append(
                {
                    "template": template,
                    "timeout": timeout,
                    "metadata": metadata,
                    "envs": envs,
                    "secure": secure,
                    "allow_internet_access": allow_internet_access,
                    "network": network,
                    "lifecycle": lifecycle,
                    "mcp": mcp,
                }
            )
            return _FakeE2BSandbox()

    monkeypatch.setattr(
        e2b_module, "_import_sandbox_class", lambda _sandbox_type: _FakeSandboxFactory
    )

    client = E2BSandboxClient()
    session = await client.create(
        options=E2BSandboxClientOptions(
            sandbox_type="e2b",
            exposed_ports=(8765,),
        )
    )

    assert create_calls
    assert create_calls[0]["network"] == {"allow_public_traffic": True}
    assert create_calls[0]["lifecycle"] == {"on_timeout": "pause", "auto_resume": True}
    assert isinstance(session.state, E2BSandboxSessionState)
    assert session.state.exposed_ports == (8765,)
    assert session.state.on_timeout == "pause"
    assert session.state.auto_resume is True


@pytest.mark.asyncio
async def test_e2b_client_create_omits_auto_resume_for_kill_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_calls: list[dict[str, object]] = []

    class _FakeSandboxFactory:
        @staticmethod
        async def create(
            *,
            template: str | None = None,
            timeout: int | None = None,
            metadata: dict[str, str] | None = None,
            envs: dict[str, str] | None = None,
            secure: bool = True,
            allow_internet_access: bool = True,
            network: dict[str, object] | None = None,
            lifecycle: dict[str, object] | None = None,
            mcp: dict[str, dict[str, str]] | None = None,
        ) -> _FakeE2BSandbox:
            _ = (
                template,
                timeout,
                metadata,
                envs,
                secure,
                allow_internet_access,
                network,
                lifecycle,
                mcp,
            )
            create_calls.append({"lifecycle": lifecycle})
            return _FakeE2BSandbox()

    monkeypatch.setattr(
        e2b_module, "_import_sandbox_class", lambda _sandbox_type: _FakeSandboxFactory
    )

    client = E2BSandboxClient()
    session = await client.create(
        options=E2BSandboxClientOptions(
            sandbox_type="e2b",
            on_timeout="kill",
        )
    )

    assert create_calls == [{"lifecycle": {"on_timeout": "kill"}}]
    assert isinstance(session.state, E2BSandboxSessionState)
    assert session.state.on_timeout == "kill"
    assert session.state.auto_resume is True


@pytest.mark.asyncio
async def test_e2b_client_create_passes_mcp_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_calls: list[dict[str, object]] = []

    class _FakeSandboxFactory:
        @staticmethod
        async def create(
            *,
            template: str | None = None,
            timeout: int | None = None,
            metadata: dict[str, str] | None = None,
            envs: dict[str, str] | None = None,
            secure: bool = True,
            allow_internet_access: bool = True,
            network: dict[str, object] | None = None,
            lifecycle: dict[str, object] | None = None,
            mcp: dict[str, dict[str, str]] | None = None,
        ) -> _FakeE2BSandbox:
            _ = (
                template,
                timeout,
                metadata,
                envs,
                secure,
                allow_internet_access,
                network,
                lifecycle,
                mcp,
            )
            create_calls.append({"mcp": mcp})
            return _FakeE2BSandbox()

    monkeypatch.setattr(
        e2b_module, "_import_sandbox_class", lambda _sandbox_type: _FakeSandboxFactory
    )

    client = E2BSandboxClient()
    await client.create(
        options=E2BSandboxClientOptions(
            sandbox_type="e2b",
            mcp={
                "exa": {"apiKey": "exa-key"},
                "browserbase": {
                    "apiKey": "browserbase-key",
                    "geminiApiKey": "gemini-key",
                    "projectId": "project-id",
                },
            },
        )
    )

    assert create_calls == [
        {
            "mcp": {
                "exa": {"apiKey": "exa-key"},
                "browserbase": {
                    "apiKey": "browserbase-key",
                    "geminiApiKey": "gemini-key",
                    "projectId": "project-id",
                },
            }
        }
    ]


def test_e2b_deserialize_session_state_defaults_missing_mcp() -> None:
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id="sb-123",
        mcp={"exa": {"apiKey": "exa-key"}},
    )
    payload = state.model_dump(mode="python")
    payload.pop("mcp")

    restored = E2BSandboxClient().deserialize_session_state(cast(dict[str, object], payload))

    assert isinstance(restored, E2BSandboxSessionState)
    assert restored.mcp is None


def test_e2b_client_options_preserves_positional_exposed_ports() -> None:
    options = E2BSandboxClientOptions(
        "e2b",
        None,
        None,
        None,
        None,
        True,
        True,
        None,
        False,
        (8765,),
    )

    assert options.exposed_ports == (8765,)
    assert options.workspace_persistence == "tar"
    assert options.on_timeout == "pause"
    assert options.auto_resume is True


@pytest.mark.asyncio
async def test_e2b_resume_reuses_paused_timeout_lifecycle_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    connected: list[tuple[str, int | None]] = []

    class _FakeSandboxFactory:
        @staticmethod
        async def create(**kwargs: object) -> _FakeE2BSandbox:
            created.append(dict(kwargs))
            return _FakeE2BSandbox()

        @staticmethod
        async def connect(*, sandbox_id: str, timeout: int | None = None) -> _FakeE2BSandbox:
            connected.append((sandbox_id, timeout))
            sandbox = _FakeE2BSandbox()
            sandbox.sandbox_id = sandbox_id
            return sandbox

    monkeypatch.setattr(
        e2b_module, "_import_sandbox_class", lambda _sandbox_type: _FakeSandboxFactory
    )

    client = E2BSandboxClient()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id="sb-paused",
        sandbox_timeout=15,
        on_timeout="pause",
        auto_resume=True,
        pause_on_exit=False,
    )

    resumed = await client.resume(state)

    assert connected == [("sb-paused", 15)]
    assert created == []
    assert isinstance(resumed.state, E2BSandboxSessionState)
    assert resumed.state.sandbox_id == "sb-paused"
    assert isinstance(resumed._inner, E2BSandboxSession)
    assert resumed._inner._workspace_state_preserved_on_start() is True  # noqa: SLF001
    assert resumed._inner._system_state_preserved_on_start() is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_resume_reuses_live_kill_timeout_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    connected: list[tuple[str, int | None]] = []

    class _LiveSandbox(_FakeE2BSandbox):
        async def is_running(self, request_timeout: float | None = None) -> bool:
            _ = request_timeout
            return True

    class _FakeSandboxFactory:
        @staticmethod
        async def create(**kwargs: object) -> _FakeE2BSandbox:
            created.append(dict(kwargs))
            return _FakeE2BSandbox()

        @staticmethod
        async def connect(*, sandbox_id: str, timeout: int | None = None) -> _LiveSandbox:
            connected.append((sandbox_id, timeout))
            sandbox = _LiveSandbox()
            sandbox.sandbox_id = sandbox_id
            return sandbox

    monkeypatch.setattr(
        e2b_module, "_import_sandbox_class", lambda _sandbox_type: _FakeSandboxFactory
    )

    client = E2BSandboxClient()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id="sb-live",
        sandbox_timeout=15,
        workspace_root_ready=True,
        on_timeout="kill",
        auto_resume=True,
        pause_on_exit=False,
    )

    resumed = await client.resume(state)

    assert connected == [("sb-live", 15)]
    assert created == []
    assert isinstance(resumed.state, E2BSandboxSessionState)
    assert resumed.state.sandbox_id == "sb-live"
    assert resumed._inner._workspace_state_preserved_on_start() is True  # noqa: SLF001
    assert resumed._inner._system_state_preserved_on_start() is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_resume_recreates_dead_kill_timeout_sandbox_and_preserves_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    connected: list[tuple[str, int | None]] = []

    class _DeadSandbox(_FakeE2BSandbox):
        async def is_running(self, request_timeout: float | None = None) -> bool:
            _ = request_timeout
            return False

    class _CreatedSandbox(_FakeE2BSandbox):
        def __init__(self) -> None:
            super().__init__()
            self.sandbox_id = "sb-recreated"

    class _FakeSandboxFactory:
        @staticmethod
        async def create(
            *,
            template: str | None = None,
            timeout: int | None = None,
            metadata: dict[str, str] | None = None,
            envs: dict[str, str] | None = None,
            secure: bool = True,
            allow_internet_access: bool = True,
            network: dict[str, object] | None = None,
            lifecycle: dict[str, object] | None = None,
            mcp: dict[str, dict[str, str]] | None = None,
        ) -> _CreatedSandbox:
            _ = (
                template,
                timeout,
                metadata,
                envs,
                secure,
                allow_internet_access,
                network,
                lifecycle,
                mcp,
            )
            created.append(
                {
                    "template": template,
                    "timeout": timeout,
                    "metadata": metadata,
                    "envs": envs,
                    "secure": secure,
                    "allow_internet_access": allow_internet_access,
                    "network": network,
                    "lifecycle": lifecycle,
                    "mcp": mcp,
                }
            )
            return _CreatedSandbox()

        @staticmethod
        async def connect(*, sandbox_id: str, timeout: int | None = None) -> _DeadSandbox:
            connected.append((sandbox_id, timeout))
            sandbox = _DeadSandbox()
            sandbox.sandbox_id = sandbox_id
            return sandbox

    monkeypatch.setattr(
        e2b_module, "_import_sandbox_class", lambda _sandbox_type: _FakeSandboxFactory
    )

    client = E2BSandboxClient()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id="sb-dead",
        sandbox_timeout=15,
        workspace_root_ready=True,
        on_timeout="kill",
        auto_resume=True,
        pause_on_exit=False,
        mcp={"exa": {"apiKey": "exa-key"}},
    )

    resumed = await client.resume(state)

    assert connected == [("sb-dead", 15)]
    assert created == [
        {
            "template": None,
            "timeout": 15,
            "metadata": None,
            "envs": None,
            "secure": True,
            "allow_internet_access": True,
            "network": None,
            "lifecycle": {"on_timeout": "kill"},
            "mcp": {"exa": {"apiKey": "exa-key"}},
        }
    ]
    assert isinstance(resumed.state, E2BSandboxSessionState)
    assert resumed.state.sandbox_id == "sb-recreated"
    assert resumed.state.workspace_root_ready is False
    assert resumed._inner._workspace_state_preserved_on_start() is False  # noqa: SLF001
    assert resumed._inner._system_state_preserved_on_start() is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_normalize_path_preserves_safe_leaf_symlink_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _sandbox = _session(workspace_root_ready=True)

    async def _fake_exec(
        *command: object,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: object | None = None,
    ) -> ExecResult:
        _ = (timeout, shell, user)
        rendered = [str(part) for part in command]
        if (
            rendered[:2] == ["sh", "-c"]
            and RESOLVE_WORKSPACE_PATH_HELPER.install_marker in rendered[2]
        ):
            return ExecResult(stdout=b"", stderr=b"", exit_code=0)
        if rendered and rendered[0] == str(RESOLVE_WORKSPACE_PATH_HELPER.install_path):
            return ExecResult(stdout=b"/workspace/target.txt", stderr=b"", exit_code=0)
        raise AssertionError(f"unexpected command: {rendered!r}")

    monkeypatch.setattr(session, "exec", _fake_exec)

    normalized = await session._validate_path_access("link.txt")  # noqa: SLF001

    assert normalized == Path("/workspace/link.txt")


@pytest.mark.asyncio
async def test_e2b_normalize_path_rejects_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _sandbox = _session(workspace_root_ready=True)

    async def _fake_exec(
        *command: object,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: object | None = None,
    ) -> ExecResult:
        _ = (timeout, shell, user)
        rendered = [str(part) for part in command]
        if (
            rendered[:2] == ["sh", "-c"]
            and RESOLVE_WORKSPACE_PATH_HELPER.install_marker in rendered[2]
        ):
            return ExecResult(stdout=b"", stderr=b"", exit_code=0)
        if rendered and rendered[0] == str(RESOLVE_WORKSPACE_PATH_HELPER.install_path):
            return ExecResult(stdout=b"", stderr=b"workspace escape", exit_code=111)
        raise AssertionError(f"unexpected command: {rendered!r}")

    monkeypatch.setattr(session, "exec", _fake_exec)

    with pytest.raises(InvalidManifestPathError, match="must not escape root"):
        await session._validate_path_access("link/secret.txt")  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_persist_workspace_raises_on_nonzero_snapshot_exit() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(stderr="tar failed", exit_code=2)

    with pytest.raises(WorkspaceArchiveReadError) as exc_info:
        await session.persist_workspace()

    assert exc_info.value.context["reason"] == "snapshot_nonzero_exit"
    assert exc_info.value.context["exit_code"] == 2
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_e2b_persist_workspace_excludes_runtime_skip_paths() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    sandbox.commands.exec_root_ready = True
    session.register_persist_workspace_skip_path(Path("logs/events.jsonl"))
    sandbox.commands.next_result = _FakeE2BResult(
        stdout=base64.b64encode(b"fake-tar-bytes").decode("ascii")
    )

    archive = await session.persist_workspace()

    assert archive.read() == b"fake-tar-bytes"
    expected_command = (
        "tar --exclude=logs/events.jsonl --exclude=./logs/events.jsonl "
        "-C /workspace -cf - . | base64 -w0"
    )
    assert sandbox.commands.calls == [
        {
            "command": expected_command,
            "timeout": session.state.timeouts.snapshot_tar_s,
            "cwd": "/",
            "envs": {},
            "user": None,
        }
    ]


@pytest.mark.asyncio
async def test_e2b_persist_workspace_native_snapshot_returns_snapshot_ref() -> None:
    session, sandbox = _session(workspace_root_ready=True)
    session.state.workspace_persistence = "snapshot"

    archive = await session.persist_workspace()

    assert archive.read() == e2b_module._encode_e2b_snapshot_ref(snapshot_id="snap-123")
    assert sandbox.commands.calls == []


@pytest.mark.asyncio
async def test_e2b_persist_workspace_native_snapshot_times_out_and_remounts_mounts() -> None:
    events: list[tuple[str, str]] = []
    mount = _RecordingMount().bind_events(events)

    class _SlowSnapshotSandbox(_FakeE2BSandbox):
        async def create_snapshot(self) -> object:
            await asyncio.sleep(0.2)
            return await super().create_snapshot()

    sandbox = _SlowSnapshotSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace", entries={"mount": mount}),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
        workspace_persistence="snapshot",
    )
    state.timeouts.snapshot_tar_s = 0.01
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(WorkspaceArchiveReadError) as exc_info:
        await session.persist_workspace()

    assert exc_info.value.context["reason"] == "native_snapshot_failed"
    assert type(exc_info.value.cause).__name__ == "TimeoutError"
    assert events == [
        ("unmount", "/workspace/mount"),
        ("mount", "/workspace/mount"),
    ]


@pytest.mark.asyncio
async def test_e2b_persist_workspace_native_snapshot_falls_back_to_tar_for_plain_skip_paths() -> (
    None
):
    session, sandbox = _session(workspace_root_ready=True)
    session.state.workspace_persistence = "snapshot"
    session.register_persist_workspace_skip_path(Path("logs/events.jsonl"))
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(
        stdout=base64.b64encode(b"fake-tar-bytes").decode("ascii")
    )

    archive = await session.persist_workspace()

    assert archive.read() == b"fake-tar-bytes"
    assert sandbox.commands.calls


@pytest.mark.asyncio
async def test_e2b_hydrate_workspace_native_snapshot_recreates_from_snapshot_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, sandbox = _session(workspace_root_ready=True)
    session.state.workspace_persistence = "snapshot"
    session.state.mcp = {"exa": {"apiKey": "exa-key"}}

    created: list[dict[str, object]] = []

    class _CreatedSandbox(_FakeE2BSandbox):
        def __init__(self) -> None:
            super().__init__()
            self.sandbox_id = "sb-from-snapshot"

    class _FakeSandboxFactory:
        @staticmethod
        async def create(**kwargs: object) -> _CreatedSandbox:
            created.append(dict(kwargs))
            return _CreatedSandbox()

    monkeypatch.setattr(
        e2b_module, "_import_sandbox_class", lambda _sandbox_type: _FakeSandboxFactory
    )

    payload = io.BytesIO(e2b_module._encode_e2b_snapshot_ref(snapshot_id="snap-123"))

    await session.hydrate_workspace(payload)

    assert created == [
        {
            "template": "snap-123",
            "timeout": session.state.sandbox_timeout,
            "metadata": session.state.metadata,
            "envs": None,
            "secure": session.state.secure,
            "allow_internet_access": session.state.allow_internet_access,
            "network": None,
            "lifecycle": {"on_timeout": "pause", "auto_resume": True},
            "mcp": {"exa": {"apiKey": "exa-key"}},
        }
    ]
    assert session.state.sandbox_id == "sb-from-snapshot"
    assert session.state.workspace_root_ready is True


@pytest.mark.asyncio
async def test_e2b_hydrate_workspace_raises_on_nonzero_extract_exit() -> None:
    session, sandbox = _session(workspace_root_ready=False)
    sandbox.commands.next_result = _FakeE2BResult(stderr="tar failed", exit_code=2)

    with pytest.raises(WorkspaceArchiveWriteError) as exc_info:
        await session.hydrate_workspace(io.BytesIO(_tar_bytes()))

    assert exc_info.value.context["reason"] == "hydrate_nonzero_exit"
    assert exc_info.value.context["exit_code"] == 2
    assert session.state.workspace_root_ready is False
    assert session._workspace_root_ready is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_persist_workspace_remounts_mounts_after_snapshot() -> None:
    mount = _RecordingMount()
    sandbox = _FakeE2BSandbox()
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(
        stdout=base64.b64encode(b"fake-tar-bytes").decode("ascii")
    )
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace", entries={"mount": mount}),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    archive = await session.persist_workspace()

    assert archive.read() == b"fake-tar-bytes"
    assert mount._unmounted_paths == [Path("/workspace/mount")]
    assert mount._mounted_paths == [Path("/workspace/mount")]


@pytest.mark.asyncio
async def test_e2b_persist_workspace_uses_nested_mount_targets_and_resolved_excludes() -> None:
    parent_mount = _RecordingMount(mount_path=Path("repo"))
    child_mount = _RecordingMount(mount_path=Path("repo/sub"))
    events: list[tuple[str, str]] = []
    sandbox = _FakeE2BSandbox()
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(
        stdout=base64.b64encode(b"fake-tar-bytes").decode("ascii")
    )
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(
            root="/workspace",
            entries={
                "parent": parent_mount.bind_events(events),
                "nested": Dir(children={"child": child_mount.bind_events(events)}),
            },
        ),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    archive = await session.persist_workspace()

    assert archive.read() == b"fake-tar-bytes"
    assert [path for kind, path in events if kind == "unmount"] == [
        "/workspace/repo/sub",
        "/workspace/repo",
    ]
    assert [path for kind, path in events if kind == "mount"] == [
        "/workspace/repo",
        "/workspace/repo/sub",
    ]
    tar_command = str(sandbox.commands.calls[-1]["command"])
    assert "--exclude=repo" in tar_command
    assert "--exclude=./repo" in tar_command
    assert "--exclude=repo/sub" in tar_command
    assert "--exclude=./repo/sub" in tar_command


@pytest.mark.asyncio
async def test_e2b_persist_workspace_remounts_prior_mounts_after_unmount_failure() -> None:
    events: list[tuple[str, str]] = []
    sandbox = _FakeE2BSandbox()
    sandbox.commands.exec_root_ready = True
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(
            root="/workspace",
            entries={
                "repo": Dir(
                    children={
                        "mount1": _RecordingMount().bind_events(events),
                        "mount2": _FailingUnmountMount().bind_events(events),
                    }
                )
            },
        ),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(WorkspaceArchiveReadError):
        await session.persist_workspace()

    assert [kind for kind, _path in events] == [
        "unmount",
        "unmount_fail",
        "mount",
    ]
    assert sandbox.commands.calls == []


@pytest.mark.asyncio
async def test_e2b_persist_workspace_keeps_remounting_and_raises_remount_error_first() -> None:
    events: list[tuple[str, str]] = []
    sandbox = _FakeE2BSandbox()
    sandbox.commands.exec_root_ready = True
    sandbox.commands.next_result = _FakeE2BResult(stderr="tar failed", exit_code=2)
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(
            root="/workspace",
            entries={
                "repo": Dir(
                    children={
                        "a": _RecordingMount().bind_events(events),
                        "b": _FailingRemountMount().bind_events(events),
                    }
                )
            },
        ),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(WorkspaceArchiveReadError) as exc_info:
        await session.persist_workspace()

    assert isinstance(exc_info.value.cause, RuntimeError)
    assert str(exc_info.value.cause) == "boom while remounting second mount"
    assert exc_info.value.context["snapshot_error_before_remount_corruption"] == {
        "message": "failed to read archive for path: /workspace",
    }
    assert [kind for kind, _path in events] == [
        "unmount",
        "unmount",
        "mount_fail",
        "mount",
    ]


@pytest.mark.asyncio
async def test_e2b_clear_workspace_root_on_resume_preserves_nested_mounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _sandbox = _session()
    session.state.manifest = Manifest(
        root="/workspace",
        entries={
            "a/b": _RecordingMount(),
        },
    )
    ls_calls: list[Path] = []
    rm_calls: list[tuple[Path, bool]] = []

    async def _fake_ls(path: Path | str) -> list[object]:
        rendered = Path(path)
        ls_calls.append(rendered)
        if rendered == Path("/workspace"):
            return [
                type("Entry", (), {"path": "/workspace/a", "kind": EntryKind.DIRECTORY})(),
                type("Entry", (), {"path": "/workspace/root.txt", "kind": EntryKind.FILE})(),
            ]
        if rendered == Path("/workspace/a"):
            return [
                type("Entry", (), {"path": "/workspace/a/b", "kind": EntryKind.DIRECTORY})(),
                type("Entry", (), {"path": "/workspace/a/local.txt", "kind": EntryKind.FILE})(),
            ]
        raise AssertionError(f"unexpected ls path: {rendered}")

    async def _fake_rm(path: Path | str, *, recursive: bool = False) -> None:
        rm_calls.append((Path(path), recursive))

    monkeypatch.setattr(session, "ls", _fake_ls)
    monkeypatch.setattr(session, "rm", _fake_rm)

    await session._clear_workspace_root_on_resume()  # noqa: SLF001

    assert ls_calls == [Path("/workspace"), Path("/workspace/a")]
    assert rm_calls == [
        (Path("/workspace/a/local.txt"), True),
        (Path("/workspace/root.txt"), True),
    ]


@pytest.mark.asyncio
async def test_e2b_pty_start_and_write_stdin() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.pty.stdin_output_chunks = [b">>> "]
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await session.pty_exec_start("python3", shell=False, tty=True, yield_time_s=0.05)

    assert started.process_id is not None
    assert b">>>" in started.output

    sandbox.pty.stdin_output_chunks = [b"10\n"]
    updated = await session.pty_write_stdin(
        session_id=started.process_id,
        chars="5 + 5\n",
        yield_time_s=0.05,
    )

    assert updated.process_id == started.process_id
    assert b"10" in updated.output
    assert sandbox.pty.handle.stdin_payloads == [b"python3\n", b"5 + 5\n"]

    await session.pty_terminate_all()


@pytest.mark.asyncio
async def test_e2b_targeted_pty_termination_leaves_other_session_registered() -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    first = await session.pty_exec_start(
        "sleep 30",
        shell=False,
        tty=True,
        yield_time_s=0,
    )
    assert first.process_id is not None
    assert sandbox.pty.on_data is not None

    async def emit_tail() -> None:
        callback = sandbox.pty.on_data
        assert callable(callback)
        result = callback(b"tail")
        if inspect.isawaitable(result):
            await result

    sandbox.pty.handle.kill_hook = emit_tail
    second_handle = _FakeE2BPtyHandle()
    second_entry = e2b_module._E2BPtyProcessEntry(  # noqa: SLF001
        handle=second_handle,
        tty=True,
    )
    second_id = 1002 if first.process_id != 1002 else 1003
    session._pty_processes[second_id] = second_entry  # noqa: SLF001
    session._reserved_pty_process_ids.add(second_id)  # noqa: SLF001

    terminated = await session.pty_terminate(first.process_id)

    assert terminated.process_id is None
    assert terminated.output == b"tail"
    assert terminated.exit_code == 0
    assert sandbox.pty.handle.kill_calls == 1
    assert sandbox.commands.group_termination_calls == [sandbox.pty.handle.pid]
    assert sandbox.commands.group_termination_users == ["root"]
    assert session._pty_processes == {second_id: second_entry}  # noqa: SLF001
    assert second_handle.kill_calls == 0
    with pytest.raises(PtySessionNotFoundError):
        await session.pty_terminate(first.process_id)

    await session.pty_terminate_all()


@pytest.mark.asyncio
async def test_e2b_group_termination_leaves_sibling_group_running() -> None:
    sandbox = _FakeE2BSandbox()
    first_handle = _FakeE2BAsyncCommandHandle(pid=4101, wait_never=True)
    second_handle = _FakeE2BAsyncCommandHandle(pid=4102, wait_never=True)
    sandbox.commands.next_async_command_handle = first_handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    first = await session.pty_exec_start("sleep", "30", shell=False, tty=False, yield_time_s=0)
    sandbox.commands.next_async_command_handle = second_handle
    second = await session.pty_exec_start("sleep", "30", shell=False, tty=False, yield_time_s=0)
    assert first.process_id is not None
    assert second.process_id is not None

    await session.pty_terminate(first.process_id)

    assert sandbox.commands.group_termination_calls == [first_handle.pid]
    assert sandbox.commands.group_termination_users == ["root"]
    assert second.process_id in session._pty_processes  # noqa: SLF001
    assert not second_handle._killed.is_set()

    await session.pty_terminate(second.process_id)


@pytest.mark.asyncio
async def test_e2b_targeted_pty_termination_failure_keeps_session_registered() -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "sleep 30",
        shell=False,
        tty=True,
        yield_time_s=0,
    )
    assert started.process_id is not None
    entry = session._pty_processes[started.process_id]  # noqa: SLF001
    sandbox.pty.handle.kill_error = RuntimeError("kill failed")

    with pytest.raises(RuntimeError, match="kill failed"):
        await session.pty_terminate(started.process_id)

    assert session._pty_processes[started.process_id] is entry  # noqa: SLF001
    assert started.process_id in session._reserved_pty_process_ids  # noqa: SLF001

    sandbox.pty.handle.kill_error = None
    await session.pty_terminate(started.process_id)


@pytest.mark.asyncio
async def test_e2b_targeted_pty_termination_serializes_with_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "python3",
        shell=False,
        tty=True,
        yield_time_s=0,
    )
    assert started.process_id is not None
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    original_send_stdin = sandbox.pty.send_stdin

    async def blocked_send_stdin(
        pid: object,
        data: bytes,
        request_timeout: float | None = None,
    ) -> None:
        write_started.set()
        await release_write.wait()
        await original_send_stdin(pid, data, request_timeout)

    monkeypatch.setattr(sandbox.pty, "send_stdin", blocked_send_stdin)
    write_task = asyncio.create_task(
        session.pty_write_stdin(
            session_id=started.process_id,
            chars="1 + 1\n",
            yield_time_s=0,
        )
    )
    await write_started.wait()
    terminate_task = asyncio.create_task(session.pty_terminate(started.process_id))
    await asyncio.sleep(0)

    assert not terminate_task.done()

    release_write.set()
    updated = await write_task
    terminated = await terminate_task

    assert updated.process_id == started.process_id
    assert terminated.process_id is None


@pytest.mark.asyncio
async def test_e2b_targeted_pty_termination_interrupts_empty_output_poll() -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "sleep 30",
        shell=False,
        tty=True,
        yield_time_s=0,
    )
    assert started.process_id is not None

    poll = asyncio.create_task(
        session.pty_write_stdin(
            session_id=started.process_id,
            chars="",
            yield_time_s=30,
        )
    )
    await asyncio.sleep(0)

    terminated = await asyncio.wait_for(
        session.pty_terminate(started.process_id),
        timeout=0.5,
    )
    polled = await poll

    assert polled.process_id == started.process_id
    assert terminated.process_id is None


@pytest.mark.asyncio
async def test_e2b_concurrent_targeted_termination_kills_once() -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "sleep 30",
        shell=False,
        tty=True,
        yield_time_s=0,
    )
    assert started.process_id is not None

    first, second = await asyncio.gather(
        session.pty_terminate(started.process_id),
        session.pty_terminate(started.process_id),
        return_exceptions=True,
    )

    assert sum(isinstance(result, PtyExecUpdate) for result in (first, second)) == 1
    assert sum(isinstance(result, PtySessionNotFoundError) for result in (first, second)) == 1
    assert sandbox.pty.handle.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_targeted_termination_racing_global_cleanup_kills_once() -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "sleep 30",
        shell=False,
        tty=True,
        yield_time_s=0,
    )
    assert started.process_id is not None
    kill_started = asyncio.Event()
    release_kill = asyncio.Event()

    async def block_kill() -> None:
        kill_started.set()
        await release_kill.wait()

    sandbox.pty.handle.kill_hook = block_kill
    global_cleanup = asyncio.create_task(session.pty_terminate_all())
    await kill_started.wait()
    targeted = asyncio.create_task(session.pty_terminate(started.process_id))
    await asyncio.sleep(0)
    release_kill.set()

    await global_cleanup
    with pytest.raises(PtySessionNotFoundError):
        await targeted
    assert sandbox.pty.handle.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_cancelled_targeted_termination_finishes_cleanup() -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "sleep 30",
        shell=False,
        tty=True,
        yield_time_s=0,
    )
    assert started.process_id is not None
    entry = session._pty_processes[started.process_id]  # noqa: SLF001
    entry.last_used = 0
    kill_output_ready = asyncio.Event()
    release_kill = asyncio.Event()
    assert sandbox.pty.on_data is not None

    async def emit_tail_then_block() -> None:
        callback = sandbox.pty.on_data
        assert callable(callback)
        result = callback(b"tail")
        if inspect.isawaitable(result):
            await result
        kill_output_ready.set()
        await release_kill.wait()

    sandbox.pty.handle.kill_hook = emit_tail_then_block
    terminate_task = asyncio.create_task(session.pty_terminate(started.process_id))
    await kill_output_ready.wait()
    assert entry.termination_pending
    for process_id in range(1_000, 1_064):
        if process_id == started.process_id:
            continue
        sibling = e2b_module._E2BPtyProcessEntry(  # noqa: SLF001
            handle=_FakeE2BPtyHandle(),
            tty=True,
        )
        sibling.last_used = float(process_id)
        session._pty_processes[process_id] = sibling  # noqa: SLF001
        session._reserved_pty_process_ids.add(process_id)  # noqa: SLF001
    pruned = session._prune_pty_processes_if_needed()  # noqa: SLF001
    assert pruned is not None
    assert pruned[1] is not entry
    assert session._pty_processes[started.process_id] is entry  # noqa: SLF001

    await session._pty_lock.acquire()  # noqa: SLF001
    release_kill.set()
    for _ in range(20):
        if entry.wait_task is not None and entry.wait_task.done():
            break
        await asyncio.sleep(0)
    terminate_task.cancel()
    terminate_task.cancel()
    session._pty_lock.release()  # noqa: SLF001

    with pytest.raises(asyncio.CancelledError):
        await terminate_task

    assert started.process_id not in session._pty_processes  # noqa: SLF001
    assert started.process_id not in session._reserved_pty_process_ids  # noqa: SLF001
    with pytest.raises(PtySessionNotFoundError):
        await session.pty_terminate(started.process_id)


@pytest.mark.asyncio
async def test_e2b_cancelled_global_termination_finishes_cleanup() -> None:
    sandbox = _FakeE2BSandbox()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def block_cleanup() -> None:
        cleanup_started.set()
        await cleanup_release.wait()

    sandbox.pty.handle.kill_hook = block_cleanup
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    await session.pty_exec_start("sleep", "30", shell=False, tty=True, yield_time_s=0)

    cleanup_task = asyncio.create_task(session.pty_terminate_all())
    await cleanup_started.wait()
    cleanup_task.cancel()
    cleanup_task.cancel()
    cleanup_release.set()

    with pytest.raises(asyncio.CancelledError):
        await cleanup_task
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_targeted_termination_fails_if_output_waiter_will_not_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    monkeypatch.setattr(state.timeouts, "cleanup_s", 0.01)
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    entry = e2b_module._E2BPtyProcessEntry(  # noqa: SLF001
        handle=sandbox.pty.handle,
        tty=True,
    )
    release_waiter = asyncio.Event()

    async def cancellation_resistant_waiter() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_waiter.wait()

    entry.wait_task = asyncio.create_task(cancellation_resistant_waiter())
    session._pty_processes = {1001: entry}  # noqa: SLF001
    session._reserved_pty_process_ids = {1001}  # noqa: SLF001

    with pytest.raises(TimeoutError, match="output waiter did not stop"):
        await session.pty_terminate(1001)

    assert session._pty_processes[1001] is entry  # noqa: SLF001
    assert entry.termination_pending
    with pytest.raises(PtySessionNotFoundError):
        await session.pty_write_stdin(
            session_id=1001,
            chars="",
            yield_time_s=0,
        )

    release_waiter.set()
    await entry.wait_task
    await session.pty_terminate(1001)


@pytest.mark.asyncio
async def test_e2b_pty_start_rejects_capacity_full_of_terminating_sessions() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.commands.group_termination_error = RuntimeError("cleanup must not run")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    session._pty_processes = {  # noqa: SLF001
        process_id: e2b_module._E2BPtyProcessEntry(  # noqa: SLF001
            handle=_FakeE2BPtyHandle(),
            tty=True,
            termination_pending=True,
        )
        for process_id in range(1_000, 1_064)
    }
    session._reserved_pty_process_ids = set(session._pty_processes)  # noqa: SLF001

    with pytest.raises(ExecTransportError) as exc_info:
        await session.pty_exec_start(
            "python3",
            shell=False,
            tty=True,
            yield_time_s=0,
        )

    assert "PTY process limit reached" in str(exc_info.value.__cause__)
    assert len(session._pty_processes) == 64  # noqa: SLF001
    assert sandbox.pty.on_data is None
    assert sandbox.commands.group_termination_users == []


@pytest.mark.asyncio
async def test_e2b_failed_prune_before_provider_launch_does_not_create_tombstone() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.commands.group_termination_error = RuntimeError("cleanup must not run")
    entries: dict[int, e2b_module._E2BPtyProcessEntry] = {}  # noqa: SLF001
    for process_id in range(1_000, 1_064):
        handle = _FakeE2BPtyHandle()
        entry = e2b_module._E2BPtyProcessEntry(  # noqa: SLF001
            handle=handle,
            tty=True,
        )
        entry.last_used = float(process_id)
        entries[process_id] = entry
    oldest = cast(_FakeE2BPtyHandle, entries[1_000].handle)
    oldest.kill_error = RuntimeError("prune failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    session._pty_processes = entries  # noqa: SLF001
    session._reserved_pty_process_ids = set(entries)  # noqa: SLF001

    with pytest.raises(ExecTransportError) as exc_info:
        await session.pty_exec_start(
            "python3",
            shell=False,
            tty=True,
            yield_time_s=0,
        )

    assert "prune failed" in str(exc_info.value.__cause__)
    assert len(session._pty_processes) == 64  # noqa: SLF001
    assert len(session._reserved_pty_process_ids) == 64  # noqa: SLF001
    assert not any(entry.termination_pending for entry in entries.values())
    assert sandbox.pty.on_data is None
    assert sandbox.commands.group_termination_users == []


@pytest.mark.asyncio
async def test_e2b_concurrent_starts_do_not_exceed_process_capacity() -> None:
    sandbox = _FakeE2BSandbox()
    prune_started = asyncio.Event()
    prune_release = asyncio.Event()
    entries = {}
    for process_id in range(1_000, 1_064):
        handle = _FakeE2BPtyHandle(wait_never=True)
        entry = e2b_module._E2BPtyProcessEntry(  # noqa: SLF001
            handle=handle,
            tty=True,
        )
        entry.last_used = float(process_id)
        entries[process_id] = entry
        if process_id == 1_000:

            async def block_first_prune() -> None:
                prune_started.set()
                await prune_release.wait()

            handle.kill_hook = block_first_prune
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    session._pty_processes = entries  # noqa: SLF001
    session._reserved_pty_process_ids = set(entries)  # noqa: SLF001

    first = asyncio.create_task(
        session.pty_exec_start("sleep", "30", shell=False, tty=True, yield_time_s=0)
    )
    await prune_started.wait()
    second = asyncio.create_task(
        session.pty_exec_start("sleep", "30", shell=False, tty=True, yield_time_s=0)
    )
    await asyncio.sleep(0)

    assert len(session._pty_processes) == 64  # noqa: SLF001
    assert sandbox.pty.on_data is None

    prune_release.set()
    await asyncio.gather(first, second)
    assert len(session._pty_processes) == 64  # noqa: SLF001
    await session.pty_terminate_all()


@pytest.mark.asyncio
async def test_e2b_cancelled_provider_launch_finishes_group_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    launch_started = asyncio.Event()
    launch_release = asyncio.Event()
    original_run = sandbox.commands.run

    async def delayed_run(
        command: str,
        background: bool | None = None,
        envs: dict[str, str] | None = None,
        user: str | None = None,
        cwd: str | None = None,
        on_stdout: object | None = None,
        on_stderr: object | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
        request_timeout: float | None = None,
    ) -> object:
        if background:
            launch_started.set()
            await launch_release.wait()
        return await original_run(
            command,
            background=background,
            envs=envs,
            user=user,
            cwd=cwd,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            stdin=stdin,
            timeout=timeout,
            request_timeout=request_timeout,
        )

    monkeypatch.setattr(sandbox.commands, "run", delayed_run)
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    start_task = asyncio.create_task(
        session.pty_exec_start("sleep", "30", shell=False, tty=False, yield_time_s=0)
    )
    await launch_started.wait()
    start_task.cancel()
    start_task.cancel()
    launch_release.set()

    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert sandbox.commands.group_termination_calls == [4242]
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_global_cleanup_waits_for_provider_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    launch_started = asyncio.Event()
    launch_release = asyncio.Event()
    original_run = sandbox.commands.run

    async def delayed_run(
        command: str,
        background: bool | None = None,
        envs: dict[str, str] | None = None,
        user: str | None = None,
        cwd: str | None = None,
        on_stdout: object | None = None,
        on_stderr: object | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
        request_timeout: float | None = None,
    ) -> object:
        if background:
            launch_started.set()
            await launch_release.wait()
        return await original_run(
            command,
            background=background,
            envs=envs,
            user=user,
            cwd=cwd,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            stdin=stdin,
            timeout=timeout,
            request_timeout=request_timeout,
        )

    monkeypatch.setattr(sandbox.commands, "run", delayed_run)
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    start_task = asyncio.create_task(
        session.pty_exec_start("sleep", "30", shell=False, tty=False, yield_time_s=0)
    )
    await launch_started.wait()
    cleanup_task = asyncio.create_task(session.pty_terminate_all())
    await asyncio.sleep(0)

    assert not cleanup_task.done()
    launch_release.set()
    await asyncio.gather(start_task, cleanup_task)

    assert sandbox.commands.group_termination_calls == [4242]
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_ambiguous_launch_failure_retains_failed_cleanup_for_retry() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_never=True)
    sandbox.commands.next_async_command_handle = handle
    sandbox.commands.background_error_after_start = TimeoutError("start response lost")
    sandbox.commands.group_termination_error = RuntimeError("cleanup failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTimeoutError, match="timed out"):
        await session.pty_exec_start("sleep", "30", shell=False, tty=False, yield_time_s=0)

    assert len(session._pty_processes) == 1  # noqa: SLF001
    retained = next(iter(session._pty_processes.values()))  # noqa: SLF001
    assert retained.termination_pending

    sandbox.commands.group_termination_error = None
    await session.pty_terminate_all()
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_ambiguous_launch_waits_for_late_process_ownership() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_never=True)
    sandbox.commands.next_async_command_handle = handle
    sandbox.commands.background_late_error = TimeoutError("start response lost")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTimeoutError):
        await session.pty_exec_start("sleep", "30", shell=False, tty=False, yield_time_s=0)

    assert sandbox.commands.group_termination_calls == [handle.pid]
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_ambiguous_tty_launch_waits_for_late_process_ownership() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.pty.handle.pid = 4301
    sandbox.pty.create_late_error = TimeoutError("start response lost")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTimeoutError):
        await session.pty_exec_start("sleep", "30", shell=False, tty=True, yield_time_s=0)

    assert sandbox.commands.group_termination_calls == [4301]
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_cancelled_initial_output_collection_finishes_group_cleanup() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.commands.next_async_command_handle = _FakeE2BAsyncCommandHandle(wait_never=True)
    sandbox.commands.group_termination_release = asyncio.Event()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    start_task = asyncio.create_task(
        session.pty_exec_start("sleep", "30", shell=False, tty=False, yield_time_s=10)
    )
    while not session._pty_processes:  # noqa: SLF001
        await asyncio.sleep(0)
    start_task.cancel()
    await sandbox.commands.group_termination_started.wait()
    start_task.cancel()
    sandbox.commands.group_termination_release.set()

    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert sandbox.commands.group_termination_calls == [4242]
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_pty_start_non_tty_uses_commands_run_in_background() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.commands.async_command_stdout_chunks = ["started\n"]
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await session.pty_exec_start("python3", shell=False, tty=False, yield_time_s=0.05)

    assert started.process_id is None
    assert b"started" in started.output
    assert len(sandbox.commands.background_calls) == 1
    call = sandbox.commands.background_calls[0]
    assert call["command"] == e2b_module._e2b_supervised_command(["python3"])
    assert call["timeout"] == float(session.state.timeouts.exec_timeout_unbounded_s)
    assert call["cwd"] == "/workspace"
    assert call["stdin"] is False
    assert call["background"] is True
    assert set(cast(dict[str, str], call["envs"])) == {e2b_module._E2B_MANAGED_PROCESS_TOKEN_ENV}


@pytest.mark.asyncio
async def test_e2b_pty_start_non_tty_wakes_when_exit_follows_last_output() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_delay_s=0.01)
    sandbox.commands.next_async_command_handle = handle
    sandbox.commands.async_command_stdout_chunks = ["started\n"]
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await asyncio.wait_for(
        session.pty_exec_start("python3", shell=False, tty=False, yield_time_s=10),
        timeout=1,
    )

    assert started.process_id is None
    assert started.exit_code == 0
    assert started.output == b"started\n"
    assert handle.wait_calls == 1
    assert handle.kill_calls == 0
    assert sandbox.commands.group_termination_calls == []
    assert session._pty_processes == {}  # noqa: SLF001
    assert session._reserved_pty_process_ids == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_legacy_relocated_command_retains_termination_on_exit() -> None:
    class _LegacyRelocatedSession(E2BSandboxSession):
        async def _managed_process_command(
            self,
            sanitized_command,
            *,
            process_token,
        ):
            _ = process_token
            return ["remote-exec", *sanitized_command]

    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_delay_s=0.01)
    sandbox.commands.next_async_command_handle = handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = _LegacyRelocatedSession.from_state(state, sandbox=sandbox)

    completed = await session.pty_exec_start("true", shell=False, tty=False, yield_time_s=10)

    assert completed.exit_code == 0
    assert sandbox.commands.group_termination_calls == [handle.pid]


@pytest.mark.asyncio
async def test_e2b_pty_start_tty_wakes_when_session_exits_after_output() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BPtyHandle(wait_never=False, wait_delay_s=0.01)
    sandbox.pty.handle = handle
    sandbox.pty.stdin_output_chunks = [b"bye\n"]
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await asyncio.wait_for(
        session.pty_exec_start("exit", shell=False, tty=True, yield_time_s=10),
        timeout=1,
    )

    assert started.process_id is None
    assert started.exit_code == 0
    assert started.output == b"bye\n"
    assert handle.stdin_payloads == [b"exit\n"]
    assert handle.wait_calls == 1
    assert handle.kill_calls == 0
    assert sandbox.commands.group_termination_calls == [handle.pid]
    assert session._pty_processes == {}  # noqa: SLF001
    assert session._reserved_pty_process_ids == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_pty_start_non_tty_wakes_on_quiet_exit() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_delay_s=0.01)
    sandbox.commands.next_async_command_handle = handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await asyncio.wait_for(
        session.pty_exec_start("true", shell=False, tty=False, yield_time_s=10),
        timeout=1,
    )

    assert started.process_id is None
    assert started.exit_code == 0
    assert started.output == b""
    assert handle.wait_calls == 1
    assert handle.kill_calls == 0


@pytest.mark.asyncio
async def test_e2b_pty_start_non_tty_wakes_on_nonzero_wait_exit() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(
        wait_delay_s=0.01,
        wait_error=_FakeE2BCommandExitException(exit_code=2),
    )
    sandbox.commands.next_async_command_handle = handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await asyncio.wait_for(
        session.pty_exec_start("false", shell=False, tty=False, yield_time_s=10),
        timeout=1,
    )

    assert started.process_id is None
    assert started.exit_code == 2
    assert started.output == b""
    assert handle.wait_calls == 1
    assert handle.kill_calls == 0


@pytest.mark.asyncio
async def test_e2b_pty_start_non_tty_exited_command_preserves_waiter() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(initial_exit_code=0, wait_until_released=True)
    sandbox.commands.next_async_command_handle = handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await asyncio.wait_for(
        session.pty_exec_start("true", shell=False, tty=False, yield_time_s=10),
        timeout=1,
    )

    assert started.process_id is None
    assert started.exit_code == 0
    assert started.output == b""
    assert handle.kill_calls == 0

    for _ in range(10):
        if handle.wait_calls:
            break
        await asyncio.sleep(0)

    assert handle.wait_calls == 1
    assert not handle.wait_cancelled

    handle.release_wait()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_e2b_pty_start_non_tty_running_command_cleans_up_waiter() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_never=True)
    sandbox.commands.next_async_command_handle = handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await session.pty_exec_start("sleep", "60", shell=False, tty=False, yield_time_s=0.01)

    assert started.process_id is not None
    assert started.exit_code is None
    assert handle.wait_calls == 1
    assert handle.kill_calls == 0

    await session.pty_terminate_all()

    assert not handle.wait_cancelled
    assert handle.kill_calls == 0
    assert sandbox.commands.group_termination_calls == [handle.pid]


@pytest.mark.asyncio
async def test_e2b_failed_group_termination_remains_retryable() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_never=True)
    sandbox.commands.next_async_command_handle = handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "sleep",
        "30",
        shell=False,
        tty=False,
        yield_time_s=0,
    )
    assert started.process_id is not None
    sandbox.commands.group_termination_error = RuntimeError("group cleanup failed")

    with pytest.raises(RuntimeError, match="group cleanup failed"):
        await session.pty_terminate(started.process_id)

    assert started.process_id in session._pty_processes  # noqa: SLF001
    sandbox.commands.group_termination_error = None
    await session.pty_terminate(started.process_id)
    assert started.process_id not in session._pty_processes  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_global_cleanup_propagates_failure_and_remains_retryable() -> None:
    sandbox = _FakeE2BSandbox()
    handle = _FakeE2BAsyncCommandHandle(wait_never=True)
    sandbox.commands.next_async_command_handle = handle
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    started = await session.pty_exec_start(
        "sleep",
        "30",
        shell=False,
        tty=False,
        yield_time_s=0,
    )
    assert started.process_id is not None
    sandbox.commands.group_termination_error = RuntimeError("group cleanup failed")

    with pytest.raises(RuntimeError, match="group cleanup failed"):
        await session.pty_terminate_all()

    assert started.process_id in session._pty_processes  # noqa: SLF001
    sandbox.commands.group_termination_error = None
    await session.pty_terminate_all()
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_pty_start_non_tty_wraps_background_run_failures() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.commands.background_error = RuntimeError("background failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTransportError) as exc_info:
        await session.pty_exec_start("python3", shell=False, tty=False)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "background failed"


@pytest.mark.asyncio
async def test_e2b_stop_terminates_live_pty_sessions() -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    started = await session.pty_exec_start("python3", shell=False, tty=True, yield_time_s=0.05)
    assert started.process_id is not None

    await session.stop()

    assert sandbox.pty.handle.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_shutdown_logs_pause_failure_and_falls_back_to_kill(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.pause_error = RuntimeError("pause failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
        pause_on_exit=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    caplog.set_level(logging.WARNING, logger=e2b_module.__name__)

    await session.shutdown()

    assert sandbox.pause_calls == 1
    assert sandbox.kill_calls == 1
    assert "Failed to pause E2B sandbox on shutdown; falling back to kill." in caplog.text


@pytest.mark.asyncio
async def test_e2b_shutdown_kills_instead_of_pausing_after_process_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
        pause_on_exit=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    async def fail_process_cleanup() -> None:
        raise RuntimeError("process cleanup failed")

    monkeypatch.setattr(session, "pty_terminate_all", fail_process_cleanup)

    with pytest.raises(RuntimeError, match="process cleanup failed"):
        await session.shutdown()

    assert sandbox.pause_calls == 0
    assert sandbox.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_stop_cleanup_failure_forces_later_shutdown_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
        pause_on_exit=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)
    cleanup_calls = 0

    async def fail_first_process_cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise RuntimeError("process cleanup failed")

    monkeypatch.setattr(session, "pty_terminate_all", fail_first_process_cleanup)

    with pytest.raises(RuntimeError, match="process cleanup failed"):
        await session.stop()
    await session.shutdown()

    assert cleanup_calls == 2
    assert sandbox.pause_calls == 0
    assert sandbox.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_shutdown_logs_kill_failure_after_pause_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.pause_error = RuntimeError("pause failed")
    sandbox.kill_error = RuntimeError("kill failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
        pause_on_exit=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    caplog.set_level(logging.WARNING, logger=e2b_module.__name__)

    await session.shutdown()

    assert sandbox.pause_calls == 1
    assert sandbox.kill_calls == 1
    assert "Failed to kill E2B sandbox after pause fallback failure." in caplog.text


@pytest.mark.asyncio
async def test_e2b_shutdown_logs_direct_kill_failure(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.kill_error = RuntimeError("kill failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
        pause_on_exit=False,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    caplog.set_level(logging.WARNING, logger=e2b_module.__name__)

    await session.shutdown()

    assert sandbox.pause_calls == 0
    assert sandbox.kill_calls == 1
    assert "Failed to kill E2B sandbox on shutdown." in caplog.text


@pytest.mark.asyncio
async def test_e2b_pty_start_wraps_startup_failures() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.pty.create_error = FileNotFoundError("missing-shell")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTransportError):
        await session.pty_exec_start("python3", shell=False, tty=True)


@pytest.mark.asyncio
async def test_e2b_pty_start_cleans_up_partially_created_session_on_failure() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.pty.send_stdin_error = RuntimeError("send failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTransportError):
        await session.pty_exec_start("python3", shell=False, tty=True)

    assert sandbox.pty.handle.kill_calls == 1


@pytest.mark.asyncio
async def test_e2b_pty_start_cleans_up_partially_created_session_on_cancellation() -> None:
    sandbox = _FakeE2BSandbox()
    sandbox.pty.send_stdin_error = asyncio.CancelledError()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(asyncio.CancelledError):
        await session.pty_exec_start("python3", shell=False, tty=True)

    assert sandbox.pty.handle.kill_calls == 1
    assert session._pty_processes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_e2b_pty_start_maps_timeout_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    timeout_error_types = e2b_module._e2b_timeout_error_types()
    if timeout_error_types:
        timeout_exc = timeout_error_types[0]
    else:

        class _FakeTimeout(Exception):
            pass

        timeout_exc = _FakeTimeout
        monkeypatch.setattr(
            e2b_module,
            "_e2b_timeout_error_types",
            lambda: (_FakeTimeout,),
        )
    sandbox.pty.create_error = timeout_exc("timed out")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTimeoutError):
        await session.pty_exec_start("python3", shell=False, tty=True, timeout=2.0)


@pytest.mark.asyncio
async def test_e2b_exec_timeout_preserves_provider_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTimeout(Exception):
        def __init__(self) -> None:
            super().__init__("context deadline exceeded")
            self.stderr = "chrome stderr"

    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    monkeypatch.setattr(
        e2b_module,
        "_e2b_timeout_error_types",
        lambda: (_FakeTimeout,),
    )

    async def _raise_timeout(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise _FakeTimeout()

    monkeypatch.setattr(e2b_module, "_sandbox_run_command", _raise_timeout)

    with pytest.raises(ExecTimeoutError) as exc_info:
        await session._exec_internal("python3", "build.py", timeout=2.0)  # noqa: SLF001

    assert exc_info.value.context["provider_error"] == "context deadline exceeded"
    assert exc_info.value.context["stderr"] == "chrome stderr"


@pytest.mark.asyncio
async def test_e2b_exec_maps_httpcore_read_timeout_to_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadTimeout(Exception):
        pass

    ReadTimeout.__module__ = "httpcore"

    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    async def _raise_timeout(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise ReadTimeout()

    monkeypatch.setattr(e2b_module, "_sandbox_run_command", _raise_timeout)

    with pytest.raises(ExecTimeoutError) as exc_info:
        await session._exec_internal("python3", "build.py", timeout=2.0)  # noqa: SLF001

    assert exc_info.value.context["reason"] == "stream_read_timeout"
    assert exc_info.value.context["provider_error"] == "ReadTimeout"


@pytest.mark.asyncio
async def test_e2b_exec_maps_missing_sandbox_not_found_to_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeNotFound(Exception):
        pass

    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    monkeypatch.setattr(
        e2b_module,
        "_e2b_non_retryable_error_types",
        lambda: (_FakeNotFound,),
    )
    monkeypatch.setattr(e2b_module, "_e2b_retryable_error_types", lambda: ())
    monkeypatch.setattr(e2b_module, "_e2b_timeout_error_types", lambda: ())

    async def _raise_not_found(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise _FakeNotFound("The sandbox was not found: request failed")

    monkeypatch.setattr(e2b_module, "_sandbox_run_command", _raise_not_found)

    with pytest.raises(ExecTransportError) as exc_info:
        await session._exec_internal("python3", "build.py", timeout=2.0)  # noqa: SLF001

    assert exc_info.value.context["provider_error"] == "The sandbox was not found: request failed"
    assert exc_info.value.context["reason"] == "_FakeNotFound"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_e2b_exec_marks_rate_limit_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRateLimit(Exception):
        pass

    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    monkeypatch.setattr(e2b_module, "_e2b_retryable_error_types", lambda: (_FakeRateLimit,))
    monkeypatch.setattr(e2b_module, "_e2b_non_retryable_error_types", lambda: ())
    monkeypatch.setattr(e2b_module, "_e2b_timeout_error_types", lambda: ())

    async def _raise_rate_limit(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise _FakeRateLimit("rate limit exceeded")

    monkeypatch.setattr(e2b_module, "_sandbox_run_command", _raise_rate_limit)

    with pytest.raises(ExecTransportError) as exc_info:
        await session._exec_internal("python3", "build.py", timeout=2.0)  # noqa: SLF001

    assert exc_info.value.context["provider_error"] == "rate limit exceeded"
    assert exc_info.value.context["reason"] == "_FakeRateLimit"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_e2b_exec_marks_deterministic_provider_errors_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeGitAuth(Exception):
        pass

    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    monkeypatch.setattr(e2b_module, "_e2b_retryable_error_types", lambda: ())
    monkeypatch.setattr(e2b_module, "_e2b_non_retryable_error_types", lambda: (_FakeGitAuth,))
    monkeypatch.setattr(e2b_module, "_e2b_timeout_error_types", lambda: ())

    async def _raise_git_auth(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise _FakeGitAuth("git authentication failed")

    monkeypatch.setattr(e2b_module, "_sandbox_run_command", _raise_git_auth)

    with pytest.raises(ExecTransportError) as exc_info:
        await session._exec_internal("python3", "build.py", timeout=2.0)  # noqa: SLF001

    assert exc_info.value.context["provider_error"] == "git authentication failed"
    assert exc_info.value.context["reason"] == "_FakeGitAuth"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_e2b_exec_transport_preserves_provider_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeE2BSandbox()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    async def _raise_transport(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        raise RuntimeError("connection closed while reading HTTP status line")

    monkeypatch.setattr(e2b_module, "_sandbox_run_command", _raise_transport)

    with pytest.raises(ExecTransportError) as exc_info:
        await session._exec_internal("python3", "build.py", timeout=2.0)  # noqa: SLF001

    assert (
        exc_info.value.context["provider_error"]
        == "connection closed while reading HTTP status line"
    )


@pytest.mark.asyncio
async def test_e2b_pty_start_maps_httpcore_read_timeout_to_timeout_error() -> None:
    class ReadTimeout(Exception):
        pass

    ReadTimeout.__module__ = "httpcore"

    sandbox = _FakeE2BSandbox()
    sandbox.pty.create_error = ReadTimeout()
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTimeoutError) as exc_info:
        await session.pty_exec_start("python3", shell=False, tty=True, timeout=2.0)

    assert exc_info.value.context["reason"] == "stream_read_timeout"
    assert exc_info.value.context["provider_error"] == "ReadTimeout"


@pytest.mark.asyncio
async def test_e2b_pty_start_maps_missing_sandbox_not_found_to_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeNotFound(Exception):
        pass

    monkeypatch.setattr(
        e2b_module,
        "_e2b_non_retryable_error_types",
        lambda: (_FakeNotFound,),
    )
    monkeypatch.setattr(e2b_module, "_e2b_retryable_error_types", lambda: ())
    monkeypatch.setattr(e2b_module, "_e2b_timeout_error_types", lambda: ())

    sandbox = _FakeE2BSandbox()
    sandbox.pty.create_error = _FakeNotFound("The sandbox was not found: request failed")
    state = E2BSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root="/workspace"),
        snapshot=NoopSnapshot(id="snapshot"),
        sandbox_id=sandbox.sandbox_id,
        workspace_root_ready=True,
    )
    session = E2BSandboxSession.from_state(state, sandbox=sandbox)

    with pytest.raises(ExecTransportError) as exc_info:
        await session.pty_exec_start("python3", shell=False, tty=True, timeout=2.0)

    assert exc_info.value.context["provider_error"] == "The sandbox was not found: request failed"
    assert exc_info.value.context["reason"] == "_FakeNotFound"
    assert exc_info.value.retryable is False
