"""
Blaxel sandbox (https://blaxel.ai) implementation.

This module provides a Blaxel-backed sandbox client/session implementation backed by
``blaxel.core.sandbox.SandboxInstance``.

The ``blaxel`` dependency is optional, so package-level exports should guard imports of this
module. Within this module, Blaxel SDK imports are lazy so users without the extra can still
import the package.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import shlex
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ....sandbox.entries import Mount
from ....sandbox.errors import (
    ExecNonZeroError,
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    InvalidManifestPathError,
    PtySessionNotFoundError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceWriteTypeError,
)
from ....sandbox.files import EntryKind, FileEntry
from ....sandbox.manifest import Manifest
from ....sandbox.session import SandboxSession, SandboxSessionState
from ....sandbox.session.base_sandbox_session import BaseSandboxSession
from ....sandbox.session.dependencies import Dependencies
from ....sandbox.session.manager import Instrumentation
from ....sandbox.session.pty_lifecycle import await_task_ignoring_cancellation
from ....sandbox.session.pty_types import (
    PTY_PROCESSES_MAX,
    PTY_PROCESSES_WARNING,
    PtyExecUpdate,
    allocate_pty_process_id,
    clamp_pty_yield_time_ms,
    process_id_to_prune_from_meta,
    resolve_pty_write_yield_time_ms,
    truncate_text_by_tokens,
)
from ....sandbox.session.runtime_helpers import RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript
from ....sandbox.session.sandbox_client import BaseSandboxClient
from ....sandbox.session.stat_script import STAT_SCRIPT
from ....sandbox.session.tar_workspace import shell_tar_exclude_args
from ....sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from ....sandbox.types import ExecResult, ExposedPortEndpoint, Permissions, User
from ....sandbox.util.retry import (
    TRANSIENT_HTTP_STATUS_CODES,
    exception_chain_contains_type,
    exception_chain_has_status_code,
    iter_exception_chain,
    retry_async,
)
from ....sandbox.util.tar_utils import UnsafeTarMemberError, validate_tar_bytes
from ....sandbox.workspace_paths import (
    coerce_posix_path,
    posix_path_as_path,
    posix_path_for_error,
    sandbox_path_str,
)

DEFAULT_BLAXEL_WORKSPACE_ROOT = "/workspace"
logger = logging.getLogger(__name__)

_BLAXEL_MANAGED_PROCESS_TOKEN_ENV = "OPENAI_AGENTS_MANAGED_PROCESS_TOKEN"
_BLAXEL_PROCESS_GROUP_MAX_START_POLLS = 20
_BLAXEL_PROCESS_GROUP_TERM_POLLS = 10
_BLAXEL_PROCESS_GROUP_TERM_POLL_S = 0.05
_BLAXEL_PROCESS_STATUS_POLL_S = 0.05

_BLAXEL_PROCESS_SUPERVISOR = f"""
import os
import signal
import subprocess
import sys
import time


def keep_supervisor_alive(_signum, _frame):
    pass


token = os.environ.get({_BLAXEL_MANAGED_PROCESS_TOKEN_ENV!r}, "").encode()
needle = {_BLAXEL_MANAGED_PROCESS_TOKEN_ENV!r}.encode() + b"=" + token


def owned_process_exists():
    own_pid = os.getpid()
    own_group = os.getpgrp()
    for item in os.scandir("/proc"):
        if not item.name.isdigit() or int(item.name) == own_pid:
            continue
        try:
            process_path = f"/proc/{{item.name}}"
            with open(f"{{process_path}}/stat", encoding="utf-8") as stream:
                stat = stream.read()
            fields = stat[stat.rfind(")") + 2:].split()
            if int(fields[2]) == own_group:
                return True
            if token:
                with open(f"{{process_path}}/environ", "rb") as stream:
                    if needle in stream.read().split(b"\\0"):
                        return True
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return False


signal.signal(signal.SIGTERM, keep_supervisor_alive)
process = subprocess.Popen(sys.argv[1:], start_new_session=True)
status = process.wait()
while owned_process_exists():
    time.sleep(0.05)
raise SystemExit(128 - status if status < 0 else status)
"""

_BLAXEL_PROCESS_GROUP_TERMINATOR = f"""
import os
import signal
import sys
import time

needle = {_BLAXEL_MANAGED_PROCESS_TOKEN_ENV!r}.encode() + b"=" + sys.argv[1].encode()


def managed_groups():
    groups = set()
    for item in os.scandir("/proc"):
        if not item.name.isdigit():
            continue
        try:
            with open(f"/proc/{{item.name}}/environ", "rb") as stream:
                if needle not in stream.read().split(b"\\0"):
                    continue
            with open(f"/proc/{{item.name}}/stat", encoding="utf-8") as stream:
                stat = stream.read()
            fields = stat[stat.rfind(")") + 2:].split()
            pgid = int(fields[2])
            if pgid > 1:
                groups.add(pgid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return groups


def signal_groups(groups, sig):
    for pgid in groups:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass


groups = set()
start_polls = int(sys.argv[2])
for attempt in range(start_polls):
    groups = managed_groups()
    if groups:
        break
    if attempt + 1 < start_polls:
        time.sleep({_BLAXEL_PROCESS_GROUP_TERM_POLL_S})

if groups:
    signal_groups(groups, signal.SIGTERM)
    for _ in range({_BLAXEL_PROCESS_GROUP_TERM_POLLS}):
        groups = managed_groups()
        if not groups:
            break
        time.sleep({_BLAXEL_PROCESS_GROUP_TERM_POLL_S})
    else:
        signal_groups(managed_groups(), signal.SIGKILL)
"""


def _blaxel_process_group_termination_command(
    process_token: str,
    *,
    start_polls: int,
) -> str:
    return shlex.join(
        ("python3", "-c", _BLAXEL_PROCESS_GROUP_TERMINATOR, process_token, str(start_polls))
    )


def _blaxel_supervised_command(command: Sequence[str | Path]) -> str:
    return shlex.join(
        ("exec", "python3", "-c", _BLAXEL_PROCESS_SUPERVISOR) + tuple(str(part) for part in command)
    )


def _blaxel_tty_command(
    command: Sequence[str | Path],
    process_token: str,
    completion_token: str,
) -> tuple[str, bytes]:
    marker = f"\x1eOPENAI_AGENTS_EXIT:{completion_token}:".encode()
    command_text = shlex.join(str(part) for part in command)
    wrapped = (
        f"env {_BLAXEL_MANAGED_PROCESS_TOKEN_ENV}={shlex.quote(process_token)} {command_text}; "
        "__openai_agents_status=$?; "
        f"printf '\\036OPENAI_AGENTS_EXIT:{completion_token}:%s\\037' "
        '"$__openai_agents_status"'
    )
    return wrapped, marker


# Blaxel documents structured API error codes and retryability at:
# https://docs.blaxel.ai/troubleshooting/error-codes
_BLAXEL_ERROR_CODE_RETRYABLE: dict[str, bool] = {
    "ROUTE_NOT_FOUND": False,  # 404
    "WORKLOAD_NOT_FOUND": False,  # 404
    "WORKSPACE_NOT_FOUND": False,  # 404
    "WORKLOAD_UNAVAILABLE": True,  # 404
    "AUTHENTICATION_REQUIRED": False,  # 401
    "AUTHENTICATION_FAILED": False,  # 401
    "FORBIDDEN": False,  # 403
    "BAD_REQUEST": False,  # 400
    "USAGE_LIMIT_EXCEEDED": False,  # 402
    "POLICY_VIOLATION": False,  # varies
}


def _coerce_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            return {str(key): item for key, item in decoded.items()}
    return None


def _blaxel_error_payload(error: BaseException) -> dict[str, object] | None:
    for candidate in iter_exception_chain(error):
        for attr in ("body", "payload"):
            payload = _coerce_mapping(getattr(candidate, attr, None))
            if payload is not None:
                return payload

        response = getattr(candidate, "response", None)
        response_json = getattr(response, "json", None)
        if callable(response_json):
            try:
                payload = _coerce_mapping(response_json())
            except Exception:
                payload = None
            if payload is not None:
                return payload

        response_text = getattr(response, "text", None)
        payload = _coerce_mapping(response_text)
        if payload is not None:
            return payload

    return None


def _blaxel_structured_error(error: BaseException) -> dict[str, object] | None:
    payload = _blaxel_error_payload(error)
    if payload is None:
        return None
    nested = payload.get("error")
    if isinstance(nested, dict):
        return {str(key): value for key, value in nested.items()}
    return payload


def _blaxel_provider_retryability(error: BaseException) -> tuple[bool | None, str | None]:
    structured_error = _blaxel_structured_error(error)
    if structured_error is not None:
        retryable = structured_error.get("retryable")
        if isinstance(retryable, bool):
            code = structured_error.get("code")
            return retryable, str(code) if isinstance(code, str) and code else None

        code = structured_error.get("code")
        if isinstance(code, str):
            return _BLAXEL_ERROR_CODE_RETRYABLE.get(code), code

    return None, None


def _blaxel_provider_error_detail(error: BaseException) -> str | None:
    message = str(error)
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int):
        if message:
            return f"HTTP {status}: {message}"
        return f"HTTP {status}"
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def _blaxel_exec_transport_error(
    *,
    command: tuple[str | Path, ...],
    cause: BaseException,
) -> ExecTransportError:
    detail = _blaxel_provider_error_detail(cause)
    context: dict[str, object] = {"backend": "blaxel"}
    retryable, provider_error_code = _blaxel_provider_retryability(cause)
    if provider_error_code is not None:
        context["provider_error_code"] = provider_error_code
    if detail:
        context["provider_error"] = detail
    status = getattr(cause, "status_code", None) or getattr(cause, "status", None)
    if isinstance(status, int):
        context["http_status"] = status
        if retryable is None and status in TRANSIENT_HTTP_STATUS_CODES:
            retryable = True
    message = "Blaxel exec failed"
    if detail:
        message = f"{message}: {detail}"
    return ExecTransportError(
        command=command,
        context=context,
        cause=cause,
        message=message,
        retryable=retryable,
    )


def _import_blaxel_sdk() -> Any:
    """Lazily import SandboxInstance from the Blaxel SDK, raising a clear error if missing."""
    try:
        from blaxel.core.sandbox import SandboxInstance

        return SandboxInstance
    except ImportError as e:
        raise ImportError(
            "BlaxelSandboxClient requires the optional `blaxel` dependency.\n"
            "Install the Blaxel extra before using this sandbox backend."
        ) from e


def _import_aiohttp() -> Any:
    """Lazily import aiohttp for WebSocket PTY support."""
    try:
        import aiohttp

        return aiohttp
    except ImportError as e:
        raise ImportError(
            "PTY support for BlaxelSandboxSession requires the `aiohttp` package.\n"
            "Install it with: pip install aiohttp"
        ) from e


def _has_aiohttp() -> bool:
    """Check whether aiohttp is available without raising."""
    try:
        import aiohttp  # noqa: F401

        return True
    except ImportError:
        return False


def _import_sandbox_api_error() -> type[BaseException] | None:
    """Best-effort import of ``SandboxAPIError`` from the Blaxel SDK.

    Returns the exception class or ``None`` if the SDK is not installed.
    ``SandboxAPIError`` carries a ``status_code`` attribute that lets us
    classify errors (e.g. 404 for not-found, 408/504 for timeouts).
    """
    try:
        from blaxel.core.sandbox import SandboxAPIError

        return cast(type[BaseException], SandboxAPIError)
    except Exception:
        return None


class BlaxelTimeouts(BaseModel):
    """Timeout configuration for Blaxel sandbox operations."""

    model_config = {"frozen": True}

    exec_timeout_s: float = Field(default=300.0, ge=1)
    cleanup_s: float = Field(default=30.0, ge=1)
    file_upload_s: float = Field(default=1800.0, ge=1)
    file_download_s: float = Field(default=1800.0, ge=1)
    workspace_tar_s: float = Field(default=300.0, ge=1)
    fast_op_s: float = Field(default=30.0, ge=1)


@dataclass(frozen=True)
class BlaxelSandboxClientOptions:
    """Client options for the Blaxel sandbox."""

    image: str | None = None
    memory: int | None = None
    region: str | None = None
    ports: tuple[dict[str, Any], ...] | None = None
    env_vars: dict[str, str] | None = None
    labels: dict[str, str] | None = None
    ttl: str | None = None
    name: str | None = None
    pause_on_exit: bool = False
    timeouts: BlaxelTimeouts | dict[str, object] | None = None
    exposed_port_public: bool = True
    exposed_port_url_ttl_s: int = 3600


class BlaxelSandboxSessionState(SandboxSessionState):
    """Serializable state for a Blaxel-backed session."""

    type: Literal["blaxel"] = "blaxel"
    sandbox_name: str
    image: str | None = None
    memory: int | None = None
    region: str | None = None
    base_env_vars: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    ttl: str | None = None
    pause_on_exit: bool = False
    timeouts: BlaxelTimeouts = Field(default_factory=BlaxelTimeouts)
    sandbox_url: str | None = None
    exposed_port_public: bool = True
    exposed_port_url_ttl_s: int = 3600


# ---------------------------------------------------------------------------
# PTY session entry
# ---------------------------------------------------------------------------


@dataclass
class _BlaxelPtySessionEntry:
    process_name: str | None = None
    ws_session_id: str | None = None
    ws: Any = None
    http_session: Any = None
    tty: bool = True
    process_token: str = field(default_factory=lambda: uuid.uuid4().hex)
    completion_marker: bytes | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    output_poll_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    termination_pending: bool = False
    output_chunks: deque[bytes] = field(default_factory=deque)
    output_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    output_notify: asyncio.Event = field(default_factory=asyncio.Event)
    terminal_buffer: bytearray = field(default_factory=bytearray)
    last_used: float = field(default_factory=time.monotonic)
    exit_code: int | None = None
    wait_task: asyncio.Task[None] | None = None
    consumed_log_bytes: int = 0
    completion_tracks_descendants: bool = False


@dataclass(frozen=True)
class _BlaxelManagedProcessCommand:
    argv: list[str]
    completion_tracks_descendants: bool


# ---------------------------------------------------------------------------
# Sandbox session
# ---------------------------------------------------------------------------


class BlaxelSandboxSession(BaseSandboxSession):
    """Blaxel-backed sandbox session implementation."""

    state: BlaxelSandboxSessionState
    _sandbox: Any  # SandboxInstance
    _token: str | None
    _pty_launch_lock: asyncio.Lock
    _pty_lock: asyncio.Lock
    _pty_sessions: dict[int, _BlaxelPtySessionEntry]
    _reserved_pty_process_ids: set[int]

    def __init__(
        self,
        *,
        state: BlaxelSandboxSessionState,
        sandbox: Any,
        token: str | None = None,
    ) -> None:
        self.state = state
        self._sandbox = sandbox
        self._token = token
        self._pty_launch_lock = asyncio.Lock()
        self._pty_lock = asyncio.Lock()
        self._pty_sessions = {}
        self._reserved_pty_process_ids = set()

    @classmethod
    def from_state(
        cls,
        state: BlaxelSandboxSessionState,
        *,
        sandbox: Any,
        token: str | None = None,
    ) -> BlaxelSandboxSession:
        return cls(state=state, sandbox=sandbox, token=token)

    @property
    def sandbox_name(self) -> str:
        return self.state.sandbox_name

    # -- exposed ports -------------------------------------------------------

    def _assert_exposed_port_configured(self, port: int) -> None:
        # Blaxel previews can be created for any port on demand; no pre-declaration needed.
        pass

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        is_public = self.state.exposed_port_public
        try:
            preview = await self._sandbox.previews.create_if_not_exists(
                {
                    "metadata": {"name": f"port-{port}"},
                    "spec": {"port": port, "public": is_public},
                }
            )
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "blaxel", "detail": "preview_creation_failed"},
                cause=e,
            ) from e

        url = _extract_preview_url(preview)
        if not isinstance(url, str) or not url:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "blaxel", "detail": "invalid_preview_url", "url": url},
            )

        # For private previews, create a time-limited token.
        query = ""
        if not is_public:
            try:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=self.state.exposed_port_url_ttl_s,
                )
                token = await preview.tokens.create(expires_at)
                token_value = getattr(token, "value", None) or getattr(token, "token", None)
                if isinstance(token_value, str) and token_value:
                    query = f"bl_preview_token={token_value}"
            except Exception as e:
                raise ExposedPortUnavailableError(
                    port=port,
                    exposed_ports=self.state.exposed_ports,
                    reason="backend_unavailable",
                    context={"backend": "blaxel", "detail": "preview_token_creation_failed"},
                    cause=e,
                ) from e

        try:
            split = urlsplit(url)
            host = split.hostname
            if host is None:
                raise ValueError("missing hostname")
            port_value = split.port or (443 if split.scheme == "https" else 80)
            return ExposedPortEndpoint(
                host=host,
                port=port_value,
                tls=split.scheme == "https",
                query=query,
            )
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "blaxel", "detail": "url_parse_failed", "url": url},
                cause=e,
            ) from e

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        # When resuming a paused sandbox, _skip_start is set by the client to
        # avoid reapplying the full manifest over files that may have changed
        # while the sandbox was paused.
        if getattr(self, "_skip_start", False):
            return

        # Ensure workspace root exists before BaseSandboxSession.start() materializes
        # the manifest.  Blaxel base images run as root and do not ship a pre-created
        # workspace directory.
        root = sandbox_path_str(self.state.manifest.root)
        try:
            await self._sandbox.process.exec(
                {
                    "command": f"mkdir -p {shlex.quote(root)}",
                    "working_dir": "/",
                    "wait_for_completion": True,
                    "timeout": 10000,
                }
            )
        except Exception as e:
            logger.debug("workspace root mkdir failed (will retry during materialization): %s", e)
        await super().start()

    async def stop(self) -> None:
        await super().stop()

    async def shutdown(self) -> None:
        await self.pty_terminate_all()
        try:
            if not self.state.pause_on_exit:
                await self._sandbox.delete()
            # When pause_on_exit is True the sandbox is kept alive.  Blaxel
            # automatically resumes it on the next connection.
        except Exception as e:
            logger.warning("sandbox delete failed during shutdown: %s", e)

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    # -- file operations -----------------------------------------------------

    async def mkdir(
        self,
        path: Path | str,
        *,
        parents: bool = False,
        user: str | User | None = None,
    ) -> None:
        if user is not None:
            path = await self._check_mkdir_with_exec(path, parents=parents, user=user)
        else:
            path = await self._validate_path_access(path, for_write=True)
        if path == Path("/"):
            return
        try:
            await self._sandbox.fs.mkdir(sandbox_path_str(path))
        except Exception as e:
            raise WorkspaceArchiveWriteError(
                path=path,
                context={"reason": "mkdir_failed"},
                cause=e,
            ) from e

    async def read(self, path: Path | str, *, user: str | User | None = None) -> io.IOBase:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            workspace_path = await self._check_read_with_exec(path, user=user)
        else:
            workspace_path = await self._validate_path_access(path)

        try:
            data: Any = await self._sandbox.fs.read_binary(sandbox_path_str(workspace_path))
            if isinstance(data, str):
                data = data.encode("utf-8")
            return io.BytesIO(bytes(data))
        except Exception as e:
            # Blaxel SDK raises ResponseError with status 404 for missing files.
            status = getattr(e, "status", None)
            if status is None and hasattr(e, "args") and e.args:
                first_arg = e.args[0]
                if isinstance(first_arg, dict):
                    status = first_arg.get("status")
            error_str = str(e).lower()
            if status == 404 or "not found" in error_str or "no such file" in error_str:
                raise WorkspaceReadNotFoundError(path=error_path, cause=e) from e
            raise WorkspaceArchiveReadError(path=error_path, cause=e) from e

    async def write(
        self,
        path: Path | str,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            await self._check_write_with_exec(path, user=user)

        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=error_path, actual_type=type(payload).__name__)

        workspace_path = await self._validate_path_access(path, for_write=True)
        try:
            await self._sandbox.fs.write_binary(sandbox_path_str(workspace_path), bytes(payload))
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=workspace_path, cause=e) from e

    # -- exec ----------------------------------------------------------------

    async def _resolved_envs(self) -> dict[str, str]:
        manifest_envs = await self.state.manifest.environment.resolve()
        return {**self.state.base_env_vars, **manifest_envs}

    def _coerce_exec_timeout(self, timeout_s: float | None) -> float:
        """Resolve the effective exec timeout in seconds."""
        if timeout_s is None:
            return float(self.state.timeouts.exec_timeout_s)
        if timeout_s <= 0:
            return 0.001
        return float(timeout_s)

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        cmd_str = shlex.join(str(c) for c in command)
        cwd = self.state.manifest.root
        exec_timeout = self._coerce_exec_timeout(timeout)
        timeout_seconds = int(max(1, math.ceil(exec_timeout)))

        # Resolve manifest + base env vars and prepend them so the executed
        # process sees them.
        envs = await self._resolved_envs()
        if envs:
            env_prefix = " ".join(f"{shlex.quote(k)}={shlex.quote(v)}" for k, v in envs.items())
            cmd_str = f"env {env_prefix} {cmd_str}"

        try:
            result = await asyncio.wait_for(
                self._sandbox.process.exec(
                    {
                        "command": cmd_str,
                        "working_dir": cwd,
                        "wait_for_completion": True,
                        "timeout": timeout_seconds,
                    }
                ),
                timeout=exec_timeout,
            )

            exit_code = int(getattr(result, "exit_code", 0) or 0)
            # Blaxel ProcessResponse uses .stdout / .stderr / .logs attributes. Prefer
            # split streams when available, and only fall back to logs/output for older SDKs.
            has_split_streams = hasattr(result, "stdout") or hasattr(result, "stderr")
            stdout = str(getattr(result, "stdout", "") or "")
            stderr = str(getattr(result, "stderr", "") or "")
            fallback = str(getattr(result, "logs", "") or getattr(result, "output", "") or "")
            stdout_bytes = stdout.encode("utf-8", errors="replace")
            stderr_bytes = stderr.encode("utf-8", errors="replace")

            if has_split_streams:
                return ExecResult(stdout=stdout_bytes, stderr=stderr_bytes, exit_code=exit_code)

            fallback_bytes = fallback.encode("utf-8", errors="replace")
            if exit_code == 0:
                return ExecResult(stdout=fallback_bytes, stderr=b"", exit_code=exit_code)
            return ExecResult(stdout=b"", stderr=fallback_bytes, exit_code=exit_code)
        except asyncio.TimeoutError as e:
            raise ExecTimeoutError(command=command, timeout_s=exec_timeout, cause=e) from e
        except (ExecTimeoutError, ExecTransportError):
            raise
        except Exception as e:
            api_error_cls = _import_sandbox_api_error()
            if api_error_cls is not None and isinstance(e, api_error_cls):
                status = getattr(e, "status_code", None)
                if status in (408, 504):
                    raise ExecTimeoutError(command=command, timeout_s=exec_timeout, cause=e) from e
            raise _blaxel_exec_transport_error(command=command, cause=e) from e

    async def stat(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> FileEntry | None:
        path_policy = self._workspace_path_policy()
        original_path = coerce_posix_path(path)
        workspace_path = path_policy.normalize_sandbox_path(path)
        path_arg = workspace_path.as_posix()
        grant_roots = tuple(
            root.as_posix() for root, _read_only in path_policy.extra_path_grant_rules()
        )
        command = ("stat", path_arg)
        result = await self.exec(
            "python3",
            "-c",
            STAT_SCRIPT,
            path_policy.sandbox_root().as_posix(),
            path_arg,
            *grant_roots,
            timeout=self.state.timeouts.fast_op_s,
            shell=False,
            user=user,
        )
        if not result.ok():
            raise ExecNonZeroError(result, command=command)

        try:
            stdout = result.stdout.decode("utf-8")
            payload = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecTransportError(
                command=command,
                context={
                    "reason": "malformed_stat_response",
                    "stdout": result.stdout.decode("utf-8", errors="replace"),
                    "stderr": result.stderr.decode("utf-8", errors="replace"),
                },
                cause=exc,
                retryable=False,
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
            raise ExecTransportError(
                command=command,
                context={"reason": "malformed_stat_response", "stdout": stdout},
                retryable=False,
            )

        status = payload["status"]
        if status == "missing":
            if payload.get("component") == path_arg:
                return None
            raise ExecTransportError(
                command=command,
                context={"reason": "malformed_stat_response", "stdout": stdout},
                retryable=False,
            )
        if status == "escape":
            resolved_path = payload.get("resolved_path")
            if not isinstance(resolved_path, str) or not resolved_path:
                raise ExecTransportError(
                    command=command,
                    context={"reason": "malformed_stat_response", "stdout": stdout},
                    retryable=False,
                )
            reason: Literal["absolute", "escape_root"] = (
                "absolute" if original_path.is_absolute() else "escape_root"
            )
            raise InvalidManifestPathError(
                rel=original_path.as_posix(),
                reason=reason,
                context={"resolved_path": resolved_path},
            )
        if status == "invalid_grant_root":
            component = payload.get("component")
            if isinstance(component, str) and component:
                raise ValueError(
                    f"extra path grant must not resolve to filesystem root: {component}"
                )
            raise ExecTransportError(
                command=command,
                context={"reason": "malformed_stat_response", "stdout": stdout},
                retryable=False,
            )
        if status in {"missing_ancestor", "not_directory"}:
            component = payload.get("component")
            if isinstance(component, str) and component:
                raise WorkspaceReadNotFoundError(
                    path=posix_path_for_error(workspace_path),
                    context={"reason": status, "component": component},
                )
            raise ExecTransportError(
                command=command,
                context={"reason": "malformed_stat_response", "stdout": stdout},
                retryable=False,
            )
        if status != "entry":
            raise ExecTransportError(
                command=command,
                context={"reason": "malformed_stat_response", "stdout": stdout},
                retryable=False,
            )

        mode = payload.get("mode")
        owner = payload.get("owner")
        group = payload.get("group")
        size = payload.get("size")
        if (
            not isinstance(mode, int)
            or isinstance(mode, bool)
            or mode < 0
            or not isinstance(owner, str)
            or not isinstance(group, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ExecTransportError(
                command=command,
                context={"reason": "malformed_stat_response", "stdout": stdout},
                retryable=False,
            )
        kind = {
            0o040000: EntryKind.DIRECTORY,
            0o100000: EntryKind.FILE,
            0o120000: EntryKind.SYMLINK,
        }.get(mode & 0o170000, EntryKind.OTHER)
        permissions = Permissions(
            owner=(mode >> 6) & 0b111,
            group=(mode >> 3) & 0b111,
            other=mode & 0b111,
            directory=kind == EntryKind.DIRECTORY,
        )
        return FileEntry(
            path=path_arg,
            permissions=permissions,
            owner=owner,
            group=group,
            size=size,
            kind=kind,
        )

    # -- running check -------------------------------------------------------

    async def running(self) -> bool:
        try:
            await asyncio.wait_for(self._sandbox.fs.ls("/"), timeout=10.0)
            return True
        except Exception as e:
            logger.debug("sandbox health check failed: %s", e)
            return False

    # -- workspace persistence -----------------------------------------------

    def _tar_exclude_args(self) -> list[str]:
        return shell_tar_exclude_args(self._persist_workspace_skip_relpaths())

    @retry_async(
        retry_if=lambda exc, self: (
            exception_chain_contains_type(exc, (asyncio.TimeoutError,))
            or exception_chain_has_status_code(exc, TRANSIENT_HTTP_STATUS_CODES)
        )
    )
    async def persist_workspace(self) -> io.IOBase:
        root = self._workspace_root_path()
        tar_path = f"/tmp/bl-persist-{self.state.session_id.hex}.tar"
        excludes = " ".join(self._tar_exclude_args())
        tar_cmd = (
            f"tar {excludes} -C {shlex.quote(root.as_posix())} -cf {shlex.quote(tar_path)} ."
        ).strip()

        unmounted_mounts: list[tuple[Mount, Path]] = []
        unmount_error: WorkspaceArchiveReadError | None = None
        for mount_entry, mount_path in self.state.manifest.ephemeral_mount_targets():
            try:
                await mount_entry.mount_strategy.teardown_for_snapshot(
                    mount_entry, self, mount_path
                )
            except Exception as e:
                unmount_error = WorkspaceArchiveReadError(path=root, cause=e)
                break
            unmounted_mounts.append((mount_entry, mount_path))

        snapshot_error: WorkspaceArchiveReadError | None = None
        raw: bytes | None = None
        if unmount_error is None:
            try:
                result = await self._exec_internal(
                    "sh", "-c", tar_cmd, timeout=self.state.timeouts.workspace_tar_s
                )
                if result.exit_code != 0:
                    raise WorkspaceArchiveReadError(
                        path=root,
                        context={
                            "reason": "tar_failed",
                            "output": result.stderr.decode("utf-8", errors="replace"),
                        },
                        retryable=False,
                    )
                raw_data: Any = await self._sandbox.fs.read_binary(tar_path)
                if isinstance(raw_data, str):
                    raw_data = raw_data.encode("utf-8")
                raw = bytes(raw_data)
            except WorkspaceArchiveReadError as e:
                snapshot_error = e
            except Exception as e:
                snapshot_error = WorkspaceArchiveReadError(path=root, cause=e)
            finally:
                try:
                    await self._exec_internal(
                        "rm", "-f", "--", tar_path, timeout=self.state.timeouts.cleanup_s
                    )
                except Exception as e:
                    logger.debug("persist cleanup rm failed (non-fatal): %s", e)

        remount_error: WorkspaceArchiveReadError | None = None
        for mount_entry, mount_path in reversed(unmounted_mounts):
            try:
                await mount_entry.mount_strategy.restore_after_snapshot(
                    mount_entry, self, mount_path
                )
            except Exception as e:
                if remount_error is None:
                    remount_error = WorkspaceArchiveReadError(path=root, cause=e)

        if remount_error is not None:
            raise remount_error
        if unmount_error is not None:
            raise unmount_error
        if snapshot_error is not None:
            raise snapshot_error

        assert raw is not None
        return io.BytesIO(raw)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        root = self._workspace_root_path()
        tar_path = f"/tmp/bl-hydrate-{self.state.session_id.hex}.tar"
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=Path(tar_path), actual_type=type(payload).__name__)

        try:
            validate_tar_bytes(
                bytes(payload),
                allow_external_symlink_targets=False,
            )
        except UnsafeTarMemberError as e:
            raise WorkspaceArchiveWriteError(
                path=root,
                context={
                    "reason": "unsafe_or_invalid_tar",
                    "member": e.member,
                    "detail": str(e),
                },
                cause=e,
            ) from e

        try:
            await self.mkdir(root, parents=True)
            await self._sandbox.fs.write_binary(tar_path, bytes(payload))
            result = await self._exec_internal(
                "sh",
                "-c",
                f"tar -C {shlex.quote(root.as_posix())} -xf {shlex.quote(tar_path)}",
                timeout=self.state.timeouts.workspace_tar_s,
            )
            if result.exit_code != 0:
                raise WorkspaceArchiveWriteError(
                    path=root,
                    context={
                        "reason": "tar_extract_failed",
                        "output": result.stderr.decode("utf-8", errors="replace"),
                    },
                )
        except WorkspaceArchiveWriteError:
            raise
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=root, cause=e) from e
        finally:
            try:
                await self._exec_internal(
                    "rm", "-f", "--", tar_path, timeout=self.state.timeouts.cleanup_s
                )
            except Exception as e:
                logger.debug("hydrate cleanup rm failed (non-fatal): %s", e)

    # -- PTY -----------------------------------------------------------------

    def supports_pty(self) -> bool:
        return self.state.sandbox_url is not None and self._token is not None and _has_aiohttp()

    async def _managed_process_command(
        self,
        sanitized_command: Sequence[str],
        *,
        process_token: str,
    ) -> _BlaxelManagedProcessCommand:
        return _BlaxelManagedProcessCommand(
            argv=[str(part) for part in sanitized_command],
            completion_tracks_descendants=True,
        )

    async def pty_exec_start(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
        tty: bool = False,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        process_token = uuid.uuid4().hex
        sanitized = self._prepare_exec_command(*command, shell=shell, user=user)
        managed_command = await self._managed_process_command(
            sanitized,
            process_token=process_token,
        )
        if isinstance(managed_command, _BlaxelManagedProcessCommand):
            sanitized = managed_command.argv
            completion_tracks_descendants = managed_command.completion_tracks_descendants
        else:
            sanitized = managed_command
            completion_tracks_descendants = False
        cwd = self.state.manifest.root
        exec_timeout = self._coerce_exec_timeout(timeout)
        process_name = None if tty else f"openai-agents-pty-{uuid.uuid4().hex}"
        ws_session_id = f"pty-{uuid.uuid4().hex}" if tty else None

        entry = _BlaxelPtySessionEntry(
            process_name=process_name,
            ws_session_id=ws_session_id,
            ws=None,
            http_session=None,
            tty=tty,
            process_token=process_token,
            completion_tracks_descendants=completion_tracks_descendants and not tty,
        )

        registered = False
        pruned_entry: tuple[int, _BlaxelPtySessionEntry] | None = None
        process_id = 0
        process_count = 0
        provider_launch_attempted = False

        async def _launch() -> None:
            nonlocal process_count, process_id, provider_launch_attempted
            nonlocal pruned_entry, registered
            async with self._pty_launch_lock:
                async with self._pty_lock:
                    process_id = allocate_pty_process_id(self._reserved_pty_process_ids)
                    self._reserved_pty_process_ids.add(process_id)
                    pruned_entry = self._prune_pty_sessions_if_needed()
                    if len(self._pty_sessions) >= PTY_PROCESSES_MAX and pruned_entry is None:
                        self._reserved_pty_process_ids.discard(process_id)
                        raise RuntimeError(
                            "PTY process limit reached while all registered sessions "
                            "are terminating"
                        )
                if pruned_entry is not None:
                    await self._retire_registered_pty_entry(
                        process_id=pruned_entry[0],
                        entry=pruned_entry[1],
                    )

                provider_launch_attempted = True
                if tty:
                    aiohttp = _import_aiohttp()
                    ws_url = _build_ws_url(
                        sandbox_url=self.state.sandbox_url or "",
                        token=self._token or "",
                        session_id=ws_session_id or "",
                        cwd=cwd,
                    )
                    entry.http_session = aiohttp.ClientSession()
                    entry.ws = await asyncio.wait_for(
                        entry.http_session.ws_connect(ws_url),
                        timeout=min(self.state.timeouts.fast_op_s, exec_timeout),
                    )
                    command_text, entry.completion_marker = _blaxel_tty_command(
                        sanitized,
                        process_token,
                        uuid.uuid4().hex,
                    )
                    entry.wait_task = asyncio.create_task(self._pty_ws_reader(entry))
                    await asyncio.wait_for(
                        entry.ws.send_str(
                            json.dumps({"type": "input", "data": command_text + "\n"})
                        ),
                        timeout=self.state.timeouts.fast_op_s,
                    )
                else:
                    envs = {
                        **(await self._resolved_envs()),
                        _BLAXEL_MANAGED_PROCESS_TOKEN_ENV: process_token,
                    }
                    await asyncio.wait_for(
                        self._sandbox.process.exec(
                            {
                                "name": process_name,
                                "command": _blaxel_supervised_command(sanitized),
                                "env": envs,
                                "working_dir": cwd,
                                "wait_for_completion": False,
                                "timeout": int(max(1, math.ceil(exec_timeout))),
                            }
                        ),
                        timeout=self.state.timeouts.fast_op_s,
                    )
                    entry.wait_task = asyncio.create_task(self._run_process_waiter(entry))

                async with self._pty_lock:
                    self._pty_sessions[process_id] = entry
                    process_count = len(self._pty_sessions)
                    registered = True

        launch_task = asyncio.create_task(_launch())
        try:
            await asyncio.shield(launch_task)
        except BaseException as error:
            await self._finish_unreturned_launch(
                launch_task=launch_task,
                entry=entry,
                registration=lambda: (registered, process_id),
                provider_launch_attempted=lambda: provider_launch_attempted,
            )
            if pruned_entry is not None:
                await await_task_ignoring_cancellation(
                    asyncio.create_task(self._cancel_pty_prune(pruned_entry))
                )
            if not registered and process_id:
                async with self._pty_lock:
                    self._reserved_pty_process_ids.discard(process_id)
            if isinstance(error, asyncio.CancelledError):
                raise
            if not isinstance(error, Exception):
                raise
            if isinstance(error, ExecTransportError):
                raise
            if isinstance(error, TimeoutError):
                raise ExecTimeoutError(
                    command=command,
                    timeout_s=exec_timeout,
                    cause=error,
                ) from error
            raise _blaxel_exec_transport_error(command=command, cause=error) from error

        try:
            if process_count >= PTY_PROCESSES_WARNING:
                logger.warning(
                    "PTY process count reached warning threshold: %s active sessions",
                    process_count,
                )

            async with entry.output_poll_lock:
                yield_time_ms = 10_000 if yield_time_s is None else int(yield_time_s * 1000)
                output, original_token_count = await self._collect_pty_output(
                    entry=entry,
                    yield_time_ms=clamp_pty_yield_time_ms(yield_time_ms),
                    max_output_tokens=max_output_tokens,
                )
                return await self._finalize_pty_update(
                    process_id=process_id,
                    entry=entry,
                    output=output,
                    original_token_count=original_token_count,
                )
        except BaseException:
            await self._finish_unreturned_registered_entry(process_id=process_id, entry=entry)
            raise

    async def pty_write_stdin(
        self,
        *,
        session_id: int,
        chars: str,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        async with self._pty_lock:
            entry = self._resolve_pty_session_entry(
                pty_processes=self._pty_sessions,
                session_id=session_id,
            )

        async with entry.operation_lock:
            async with self._pty_lock:
                if self._pty_sessions.get(session_id) is not entry:
                    raise PtySessionNotFoundError(session_id=session_id)
                if entry.termination_pending:
                    raise PtySessionNotFoundError(session_id=session_id)

            if chars:
                if not entry.tty:
                    raise RuntimeError("stdin is not available for this process")
                await asyncio.wait_for(
                    entry.ws.send_str(json.dumps({"type": "input", "data": chars})),
                    timeout=self.state.timeouts.fast_op_s,
                )
                await asyncio.sleep(0.1)

        async with entry.output_poll_lock:
            yield_time_ms = 250 if yield_time_s is None else int(yield_time_s * 1000)
            output, original_token_count = await self._collect_pty_output(
                entry=entry,
                yield_time_ms=resolve_pty_write_yield_time_ms(
                    yield_time_ms=yield_time_ms, input_empty=chars == ""
                ),
                max_output_tokens=max_output_tokens,
            )
            entry.last_used = time.monotonic()
            return await self._finalize_pty_update(
                process_id=session_id,
                entry=entry,
                output=output,
                original_token_count=original_token_count,
            )

    async def pty_terminate(self, session_id: int) -> PtyExecUpdate:
        async with self._pty_lock:
            entry = self._resolve_pty_session_entry(
                pty_processes=self._pty_sessions,
                session_id=session_id,
            )

        retirement_task = asyncio.create_task(
            self._retire_registered_pty_entry(process_id=session_id, entry=entry)
        )
        try:
            return await asyncio.shield(retirement_task)
        except asyncio.CancelledError:
            try:
                await await_task_ignoring_cancellation(retirement_task)
            except BaseException as cleanup_error:
                logger.warning(
                    "Blaxel targeted PTY cleanup failed after caller cancellation.",
                    extra={"process_id": session_id},
                    exc_info=cleanup_error,
                )
            raise

    async def _retire_registered_pty_entry(
        self,
        *,
        process_id: int,
        entry: _BlaxelPtySessionEntry,
    ) -> PtyExecUpdate:
        async with entry.operation_lock:
            async with self._pty_lock:
                if self._pty_sessions.get(process_id) is not entry:
                    raise PtySessionNotFoundError(session_id=process_id)
                entry.termination_pending = True

            await self._terminate_pty_entry(entry, best_effort=False)
            async with entry.output_poll_lock:
                output, original_token_count = await self._collect_pty_output(
                    entry=entry,
                    yield_time_ms=0,
                    max_output_tokens=None,
                )
                async with self._pty_lock:
                    if self._pty_sessions.get(process_id) is not entry:
                        raise PtySessionNotFoundError(session_id=process_id)
                    self._pty_sessions.pop(process_id)
                    self._reserved_pty_process_ids.discard(process_id)

        return PtyExecUpdate(
            process_id=None,
            output=output,
            exit_code=self._entry_exit_code(entry),
            original_token_count=original_token_count,
        )

    async def _finish_unreturned_launch(
        self,
        *,
        launch_task: asyncio.Task[None],
        entry: _BlaxelPtySessionEntry,
        registration: Callable[[], tuple[bool, int]],
        provider_launch_attempted: Callable[[], bool],
    ) -> None:
        if not launch_task.done():
            try:
                await await_task_ignoring_cancellation(launch_task)
            except BaseException:
                pass

        registered, process_id = registration()
        if registered:
            await self._finish_unreturned_registered_entry(process_id=process_id, entry=entry)
            return
        if not provider_launch_attempted():
            return

        cleanup_task = asyncio.create_task(
            self._terminate_pty_entry(
                entry,
                best_effort=False,
                reconcile_late_start=True,
            )
        )
        try:
            await await_task_ignoring_cancellation(cleanup_task)
        except BaseException as cleanup_error:
            retention_task = asyncio.create_task(self._retain_unreturned_pty_entry(entry))
            process_id = await await_task_ignoring_cancellation(retention_task)
            logger.warning(
                "Blaxel PTY cleanup failed before process registration.",
                extra={"process_id": process_id},
                exc_info=cleanup_error,
            )

    async def _retain_unreturned_pty_entry(self, entry: _BlaxelPtySessionEntry) -> int:
        async with self._pty_lock:
            process_id = allocate_pty_process_id(self._reserved_pty_process_ids)
            self._reserved_pty_process_ids.add(process_id)
            entry.termination_pending = True
            self._pty_sessions[process_id] = entry
            return process_id

    async def _finish_unreturned_registered_entry(
        self,
        *,
        process_id: int,
        entry: _BlaxelPtySessionEntry,
    ) -> None:
        cleanup_task = asyncio.create_task(
            self._retire_registered_pty_entry(process_id=process_id, entry=entry)
        )
        try:
            await await_task_ignoring_cancellation(cleanup_task)
        except BaseException as cleanup_error:
            logger.warning(
                "Blaxel PTY cleanup failed before its process ID was returned.",
                extra={"process_id": process_id},
                exc_info=cleanup_error,
            )

    async def pty_terminate_all(self) -> None:
        cleanup_task = asyncio.create_task(self._terminate_all_pty_entries())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as cancellation:
            try:
                await await_task_ignoring_cancellation(cleanup_task)
            except BaseException as cleanup_error:
                logger.warning(
                    "Blaxel PTY cleanup failed after caller cancellation.",
                    exc_info=cleanup_error,
                )
            raise cancellation

    async def _terminate_all_pty_entries(self) -> None:
        async with self._pty_launch_lock:
            async with self._pty_lock:
                entries = list(self._pty_sessions.items())

            first_error: Exception | None = None
            for process_id, entry in entries:
                try:
                    await self._retire_registered_pty_entry(process_id=process_id, entry=entry)
                except PtySessionNotFoundError:
                    continue
                except Exception as error:
                    first_error = first_error or error
                    logger.warning(
                        "Failed to terminate Blaxel PTY process during session cleanup.",
                        extra={"process_id": process_id},
                        exc_info=error,
                    )
            if first_error is not None:
                raise first_error

    # -- PTY internals -------------------------------------------------------

    async def _pty_ws_reader(self, entry: _BlaxelPtySessionEntry) -> None:
        try:
            aiohttp = _import_aiohttp()
            async for msg in entry.ws:
                if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    try:
                        raw_text = (
                            msg.data
                            if isinstance(msg.data, str)
                            else msg.data.decode("utf-8", errors="replace")
                        )
                        data = json.loads(raw_text)
                        msg_type = data.get("type", "") or data.get("Type", "")
                        if msg_type == "output":
                            raw = (data.get("data", "") or data.get("Data", "")).encode(
                                "utf-8", errors="replace"
                            )
                            visible, exit_code = self._consume_terminal_output(entry, raw)
                            await self._append_pty_output(entry, visible)
                            if exit_code is not None:
                                entry.exit_code = exit_code
                                break
                        elif msg_type == "error":
                            raw = (data.get("data", "") or data.get("Data", "")).encode(
                                "utf-8", errors="replace"
                            )
                            await self._append_pty_output(entry, raw)
                            entry.exit_code = 1
                            break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.debug("PTY ws reader: ignoring malformed message")
                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
        except Exception as e:
            logger.debug("PTY ws reader terminated with error: %s", e)
        finally:
            if entry.terminal_buffer:
                await self._append_pty_output(entry, bytes(entry.terminal_buffer))
                entry.terminal_buffer.clear()
            if entry.exit_code is None and not entry.termination_pending:
                entry.exit_code = 1
            entry.output_notify.set()

    async def _run_process_waiter(self, entry: _BlaxelPtySessionEntry) -> None:
        assert entry.process_name is not None
        try:
            while True:
                process = await self._sandbox.process.get(entry.process_name)
                logs = str(getattr(process, "logs", "") or "")
                encoded_logs = logs.encode("utf-8", errors="replace")
                if len(encoded_logs) > entry.consumed_log_bytes:
                    await self._append_pty_output(
                        entry,
                        encoded_logs[entry.consumed_log_bytes :],
                    )
                    entry.consumed_log_bytes = len(encoded_logs)

                status = str(getattr(process, "status", "running") or "running").lower()
                if status != "running":
                    value = getattr(process, "exit_code", None)
                    entry.exit_code = int(value) if value is not None else 1
                    break
                await asyncio.sleep(_BLAXEL_PROCESS_STATUS_POLL_S)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not entry.termination_pending:
                logger.debug("Blaxel process waiter failed: %s", error)
                entry.exit_code = 1
        finally:
            entry.output_notify.set()

    async def _append_pty_output(
        self,
        entry: _BlaxelPtySessionEntry,
        payload: bytes,
    ) -> None:
        if not payload:
            return
        async with entry.output_lock:
            entry.output_chunks.append(payload)
        entry.output_notify.set()

    def _consume_terminal_output(
        self,
        entry: _BlaxelPtySessionEntry,
        payload: bytes,
    ) -> tuple[bytes, int | None]:
        marker = entry.completion_marker
        if marker is None:
            return payload, None

        entry.terminal_buffer.extend(payload)
        marker_index = entry.terminal_buffer.find(marker)
        if marker_index >= 0:
            suffix_index = entry.terminal_buffer.find(b"\x1f", marker_index + len(marker))
            if suffix_index < 0:
                visible = bytes(entry.terminal_buffer[:marker_index])
                del entry.terminal_buffer[:marker_index]
                return visible, None
            visible = bytes(entry.terminal_buffer[:marker_index])
            exit_code_bytes = bytes(
                entry.terminal_buffer[marker_index + len(marker) : suffix_index]
            )
            entry.terminal_buffer.clear()
            try:
                return visible, int(exit_code_bytes.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                return visible, 1

        keep = 0
        for size in range(min(len(entry.terminal_buffer), len(marker) - 1), 0, -1):
            if entry.terminal_buffer[-size:] == marker[:size]:
                keep = size
                break
        visible_size = len(entry.terminal_buffer) - keep
        if visible_size <= 0:
            return b"", None
        visible = bytes(entry.terminal_buffer[:visible_size])
        del entry.terminal_buffer[:visible_size]
        return visible, None

    async def _collect_pty_output(
        self,
        *,
        entry: _BlaxelPtySessionEntry,
        yield_time_ms: int,
        max_output_tokens: int | None,
    ) -> tuple[bytes, int | None]:
        deadline = time.monotonic() + (yield_time_ms / 1000)
        output = bytearray()

        while True:
            async with entry.output_lock:
                while entry.output_chunks:
                    output.extend(entry.output_chunks.popleft())

            if time.monotonic() >= deadline:
                break
            if self._entry_exit_code(entry) is not None:
                async with entry.output_lock:
                    while entry.output_chunks:
                        output.extend(entry.output_chunks.popleft())
                break

            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            try:
                await asyncio.wait_for(entry.output_notify.wait(), timeout=remaining_s)
            except asyncio.TimeoutError:
                break
            entry.output_notify.clear()

        text = output.decode("utf-8", errors="replace")
        truncated_text, original_token_count = truncate_text_by_tokens(text, max_output_tokens)
        return truncated_text.encode("utf-8", errors="replace"), original_token_count

    async def _finalize_pty_update(
        self,
        *,
        process_id: int,
        entry: _BlaxelPtySessionEntry,
        output: bytes,
        original_token_count: int | None,
    ) -> PtyExecUpdate:
        exit_code = self._entry_exit_code(entry)
        live_process_id: int | None = process_id

        if exit_code is not None and not entry.termination_pending:
            async with self._pty_lock:
                if self._pty_sessions.get(process_id) is not entry:
                    return PtyExecUpdate(
                        process_id=None,
                        output=output,
                        exit_code=exit_code,
                        original_token_count=original_token_count,
                    )
                entry.termination_pending = True
            if not entry.completion_tracks_descendants:
                await self._terminate_pty_entry(entry, best_effort=False)
            async with self._pty_lock:
                if self._pty_sessions.get(process_id) is entry:
                    self._pty_sessions.pop(process_id)
                    self._reserved_pty_process_ids.discard(process_id)
            live_process_id = None
        else:
            async with self._pty_lock:
                if self._pty_sessions.get(process_id) is not entry:
                    live_process_id = None

        return PtyExecUpdate(
            process_id=live_process_id,
            output=output,
            exit_code=exit_code,
            original_token_count=original_token_count,
        )

    def _prune_pty_sessions_if_needed(
        self,
    ) -> tuple[int, _BlaxelPtySessionEntry] | None:
        if len(self._pty_sessions) < PTY_PROCESSES_MAX:
            return None
        meta: list[tuple[int, float, bool]] = [
            (process_id, entry.last_used, self._entry_exit_code(entry) is not None)
            for process_id, entry in self._pty_sessions.items()
            if not entry.termination_pending
        ]
        process_id = process_id_to_prune_from_meta(meta)
        if process_id is None:
            return None
        entry = self._pty_sessions[process_id]
        entry.termination_pending = True
        return process_id, entry

    async def _cancel_pty_prune(
        self,
        pruned: tuple[int, _BlaxelPtySessionEntry],
    ) -> None:
        process_id, entry = pruned
        async with self._pty_lock:
            if self._pty_sessions.get(process_id) is entry:
                entry.termination_pending = False

    def _entry_exit_code(self, entry: _BlaxelPtySessionEntry) -> int | None:
        if entry.exit_code is None:
            return None
        try:
            return int(entry.exit_code)
        except (TypeError, ValueError):
            return None

    async def _terminate_pty_entry(
        self,
        entry: _BlaxelPtySessionEntry,
        *,
        best_effort: bool = True,
        reconcile_late_start: bool = False,
    ) -> None:
        termination_error: Exception | None = None
        try:
            await self._terminate_blaxel_process_group(
                entry.process_token,
                reconcile_late_start=reconcile_late_start,
            )
        except Exception as error:
            termination_error = error

        if entry.process_name is not None:
            try:
                await asyncio.wait_for(
                    self._sandbox.process.kill(entry.process_name),
                    timeout=self.state.timeouts.cleanup_s,
                )
            except Exception as error:
                if not self._provider_process_is_absent(error):
                    termination_error = termination_error or error

        try:
            await self._close_pty_transport(entry)
        except Exception as error:
            termination_error = termination_error or error

        if entry.exit_code is None:
            entry.exit_code = 137
        entry.output_notify.set()

        if termination_error is not None and not best_effort:
            raise termination_error

        wait_task = entry.wait_task
        if wait_task is not None:
            if best_effort and not wait_task.done():
                wait_task.cancel()
            if best_effort:
                await asyncio.gather(wait_task, return_exceptions=True)
            else:
                await self._await_pty_waiter(entry)

    async def _terminate_blaxel_process_group(
        self,
        process_token: str,
        *,
        reconcile_late_start: bool,
    ) -> None:
        start_polls = 1
        if reconcile_late_start:
            available_s = min(
                (_BLAXEL_PROCESS_GROUP_MAX_START_POLLS - 1) * _BLAXEL_PROCESS_GROUP_TERM_POLL_S,
                max(0.0, self.state.timeouts.cleanup_s - 0.5),
            )
            start_polls += int(available_s / _BLAXEL_PROCESS_GROUP_TERM_POLL_S)
        await asyncio.wait_for(
            self._sandbox.process.exec(
                {
                    "name": f"openai-agents-cleanup-{uuid.uuid4().hex}",
                    "command": _blaxel_process_group_termination_command(
                        process_token,
                        start_polls=start_polls,
                    ),
                    "working_dir": "/",
                    "wait_for_completion": True,
                    "timeout": int(max(1, math.ceil(self.state.timeouts.cleanup_s))),
                }
            ),
            timeout=self.state.timeouts.cleanup_s,
        )

    async def _close_pty_transport(self, entry: _BlaxelPtySessionEntry) -> None:
        if entry.ws is not None:
            await entry.ws.close()
        if entry.http_session is not None:
            await entry.http_session.close()

    async def _await_pty_waiter(self, entry: _BlaxelPtySessionEntry) -> None:
        wait_task = entry.wait_task
        if wait_task is None:
            return
        _, pending = await asyncio.wait({wait_task}, timeout=self.state.timeouts.cleanup_s)
        if not pending:
            return

        wait_task.cancel()
        _, pending = await asyncio.wait({wait_task}, timeout=self.state.timeouts.cleanup_s)
        if pending:
            raise TimeoutError("Blaxel PTY output waiter did not stop after termination")

    def _provider_process_is_absent(self, error: Exception) -> bool:
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        error_text = str(error).lower()
        return status == 404 or "not found" in error_text


# ---------------------------------------------------------------------------
# Sandbox client
# ---------------------------------------------------------------------------


class BlaxelSandboxClient(BaseSandboxClient["BlaxelSandboxClientOptions"]):
    """Blaxel sandbox client managing sandbox lifecycle via the Blaxel SDK."""

    backend_id = "blaxel"
    _instrumentation: Instrumentation
    _token: str | None

    def __init__(
        self,
        *,
        token: str | None = None,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        # Validate that the Blaxel SDK is importable.
        _import_blaxel_sdk()
        self._instrumentation = instrumentation or Instrumentation()
        self._dependencies = dependencies
        self._token = token or os.environ.get("BL_API_KEY")

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: BlaxelSandboxClientOptions,
    ) -> SandboxSession:
        if manifest is None:
            manifest = Manifest(root=DEFAULT_BLAXEL_WORKSPACE_ROOT)

        timeouts_in = options.timeouts
        if isinstance(timeouts_in, BlaxelTimeouts):
            timeouts = timeouts_in
        elif timeouts_in is None:
            timeouts = BlaxelTimeouts()
        else:
            timeouts = BlaxelTimeouts.model_validate(timeouts_in)

        session_id = uuid.uuid4()
        sandbox_name = options.name or f"agents-{session_id.hex[:12]}"

        SandboxInstance = _import_blaxel_sdk()
        create_config = _build_create_config(
            name=sandbox_name,
            image=options.image,
            memory=options.memory,
            region=options.region,
            ports=options.ports,
            env_vars=options.env_vars,
            labels=options.labels,
            ttl=options.ttl,
            manifest=manifest,
        )
        blaxel_sandbox = await SandboxInstance.create_if_not_exists(create_config)

        sandbox_url = _get_sandbox_url(blaxel_sandbox)
        snapshot_instance = resolve_snapshot(snapshot, str(session_id))
        state = BlaxelSandboxSessionState(
            session_id=session_id,
            manifest=manifest,
            snapshot=snapshot_instance,
            sandbox_name=sandbox_name,
            image=options.image,
            memory=options.memory,
            region=options.region,
            base_env_vars=dict(options.env_vars or {}),
            labels=dict(options.labels or {}),
            ttl=options.ttl,
            pause_on_exit=options.pause_on_exit,
            timeouts=timeouts,
            sandbox_url=sandbox_url,
            exposed_port_public=options.exposed_port_public,
            exposed_port_url_ttl_s=options.exposed_port_url_ttl_s,
        )
        inner = BlaxelSandboxSession.from_state(state, sandbox=blaxel_sandbox, token=self._token)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def close(self) -> None:
        """No persistent HTTP client to close; provided for API symmetry."""

    async def __aenter__(self) -> BlaxelSandboxClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, BlaxelSandboxSession):
            raise TypeError("BlaxelSandboxClient.delete expects a BlaxelSandboxSession")
        try:
            await inner.shutdown()
        except Exception as e:
            logger.warning("shutdown error during delete (non-fatal): %s", e)
        return session

    async def resume(
        self,
        state: SandboxSessionState,
    ) -> SandboxSession:
        """Resume a sandbox from persisted state.

        When ``pause_on_exit`` is set, Blaxel automatically resumes the paused
        sandbox on connection -- this method simply reconnects by sandbox name
        via ``SandboxInstance.get()``.  If the sandbox is no longer available
        (e.g. it expired), a fresh one is created with the same configuration.
        """
        if not isinstance(state, BlaxelSandboxSessionState):
            raise TypeError("BlaxelSandboxClient.resume expects a BlaxelSandboxSessionState")

        SandboxInstance = _import_blaxel_sdk()
        blaxel_sandbox = None
        reconnected = False

        if state.pause_on_exit:
            try:
                blaxel_sandbox = await SandboxInstance.get(state.sandbox_name)
                reconnected = True
            except Exception as e:
                logger.debug("sandbox get() failed, will recreate: %s", e)

        if not reconnected or blaxel_sandbox is None:
            create_config = _build_create_config(
                name=state.sandbox_name,
                image=state.image,
                memory=state.memory,
                region=state.region,
                env_vars=state.base_env_vars or None,
                labels=state.labels or None,
                ttl=state.ttl,
            )
            blaxel_sandbox = await SandboxInstance.create_if_not_exists(create_config)

        sandbox_url = _get_sandbox_url(blaxel_sandbox)
        if sandbox_url:
            state.sandbox_url = sandbox_url

        inner = BlaxelSandboxSession.from_state(state, sandbox=blaxel_sandbox, token=self._token)
        if state.pause_on_exit and reconnected:
            inner._skip_start = True  # type: ignore[attr-defined]
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return BlaxelSandboxSessionState.model_validate(payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_create_config(
    *,
    name: str,
    image: str | None = None,
    memory: int | None = None,
    region: str | None = None,
    ports: tuple[dict[str, Any], ...] | None = None,
    env_vars: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    ttl: str | None = None,
    manifest: Manifest | None = None,
) -> dict[str, Any]:
    """Build the dict config accepted by ``SandboxInstance.create_if_not_exists``."""
    config: dict[str, Any] = {"name": name}

    if image:
        config["image"] = image
    if memory is not None:
        config["memory"] = memory
    resolved_region = region or os.environ.get("BL_REGION") or "us-pdx-1"
    config["region"] = resolved_region
    if labels:
        config["labels"] = labels
    if ttl:
        config["ttl"] = ttl

    # Pass base env vars for sandbox creation.  The session will re-resolve
    # manifest environment variables at exec time.
    all_envs: dict[str, str] = {}
    if env_vars:
        all_envs.update(env_vars)
    if all_envs:
        config["envs"] = [{"name": k, "value": v} for k, v in all_envs.items()]

    if ports:
        config["ports"] = list(ports)

    return config


def _get_sandbox_url(sandbox_instance: Any) -> str | None:
    """Best-effort extract the sandbox URL from a SandboxInstance."""
    # Try sandbox_instance.sandbox.metadata.url (standard path).
    sandbox_model = getattr(sandbox_instance, "sandbox", None)
    if sandbox_model is not None:
        metadata = getattr(sandbox_model, "metadata", None)
        if metadata is not None:
            url = getattr(metadata, "url", None)
            if isinstance(url, str) and url:
                return url
    # Try direct .url attribute.
    url = getattr(sandbox_instance, "url", None)
    if isinstance(url, str) and url:
        return url
    return None


def _extract_preview_url(preview: Any) -> str | None:
    """Extract URL string from a preview object, trying several attribute paths.

    Blaxel SDK returns a ``SandboxPreview`` whose URL lives at ``preview.spec.url``.
    """
    # Try spec.url first (Blaxel SDK path).
    for nested in ("spec", "status"):
        obj = getattr(preview, nested, None)
        if obj is not None:
            val = getattr(obj, "url", None)
            if isinstance(val, str) and val:
                return val
    # Try direct attributes.
    for attr in ("url", "endpoint"):
        val = getattr(preview, attr, None)
        if isinstance(val, str) and val:
            return val
    # Try the nested .preview.spec.url path.
    inner = getattr(preview, "preview", None)
    if inner is not None:
        return _extract_preview_url(inner)
    return None


def _build_ws_url(
    *,
    sandbox_url: str,
    token: str,
    session_id: str,
    cwd: str,
    cols: int = 80,
    rows: int = 24,
) -> str:
    """Build the WebSocket URL for a Blaxel terminal session."""
    base = sandbox_url.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    return (
        f"{ws_base}/terminal/ws"
        f"?token={token}"
        f"&cols={cols}"
        f"&rows={rows}"
        f"&sessionId={session_id}"
        f"&workingDir={cwd}"
    )


__all__ = [
    "DEFAULT_BLAXEL_WORKSPACE_ROOT",
    "BlaxelSandboxClient",
    "BlaxelSandboxClientOptions",
    "BlaxelSandboxSession",
    "BlaxelSandboxSessionState",
    "BlaxelTimeouts",
]
