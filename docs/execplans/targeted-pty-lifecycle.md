# Make targeted PTY lifecycle ownership complete

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log,
and Outcomes & Retrospective must stay up to date as work proceeds.

Maintain this document in accordance with `PLANS.md`.

## Purpose / Big Picture

The fork's additive `BaseSandboxSession.pty_terminate(session_id)` API must safely support a
caller that starts one managed process, enforces its own monotonic deadline, and stops that process
without affecting unrelated sandbox work. After this change, cancellation during provider launch
or initial output collection cannot orphan a registered process, and the E2B and Daytona adapters
provide the lifecycle behavior Feather relies on.

## Progress

- [x] (2026-07-27) Reproduced the launch-deadline gap and traced E2B termination through the SDK
  and E2B envd.
- [x] (2026-07-27) Confirmed the compatibility boundary: `pty_terminate` is unreleased fork-only
  API behavior, so no compatibility shim is required.
- [x] (2026-07-27) Finalized the provider-specific ownership design: E2B commands carry a unique
  ownership token and non-TTY commands run through a token-bearing process-group supervisor;
  Daytona retains deterministic provider-session ownership.
- [x] (2026-07-27) Implemented cancellation-safe provider launch, registration, initial-output
  collection, targeted termination, pruning, and global cleanup for E2B and Daytona.
- [x] (2026-07-27) Added launch barriers, pre-launch capacity retirement, E2B late-token
  reconciliation, and bounded Daytona late-create reconciliation.
- [x] (2026-07-27) Added focused tests for descendants, sibling isolation, repeated
  cancellation, retryable cleanup, terminal output, and provider launch failure.
- [x] (2026-07-27) Retained E2B processes whose cleanup fails before local registration so later
  session cleanup can retry them.
- [x] (2026-07-27) Ran the focused E2B, Daytona, and base-session suite (198 passed, 2
  Linux-only process tests skipped on macOS).
- [x] (2026-07-27) Ran the repository's complete format, lint, type-check, and test verification
  script on the final tree.

## Surprises & Discoveries

- Observation: E2B's command handle kill path signals only the tracked command PID.
  Evidence: E2B envd `Handler.SendSignal` calls `p.cmd.Process.Signal(signal)` rather than
  signaling a process group.
- Observation: both adapters register a process before their initial bounded output collection,
  but cancellation during that collection is outside the existing pre-registration cleanup path.
  Evidence: `pty_exec_start` leaves its `try` block before acquiring `output_poll_lock`.
- Observation: the official public documentation does not specify PTY descendant semantics; the
  local SDK and provider implementations are the authoritative contract for this fork.
- Observation: a raw numeric PGID is not a safe retry identity after an ambiguous provider
  response, and placing identity only in the user command's environment fails for `env -i`.
  Evidence: a delayed cleanup can observe PID reuse, while environment-scrubbing commands remove
  inherited variables from the user process.
- Observation: registering a Daytona placeholder before provider creation is safe only while the
  launch holds the entry's operation lock.
  Evidence: without that lock, concurrent session cleanup can delete the not-yet-created provider
  session, remove the registry entry, and allow the launch request to create an unowned session
  afterward.
- Observation: the existing pruning path removed entries before fallible provider cleanup.
  Evidence: a provider cleanup failure discarded the local process ID, so neither targeted nor
  global cleanup could retry it.
- Observation: E2B can create a provider process before validating the provider PID or obtaining
  registry capacity.
  Evidence: a failure in either step previously left no registry entry if immediate cleanup also
  failed.
- Observation: local subprocess harnesses retain a child handle before cancellation can race it;
  remote providers have a pre-handle ambiguity when a create request commits after its response is
  lost.
  Evidence: Claude Code, Qwen Code, and OpenCode terminate retained child handles/controllers,
  while E2B and Daytona require token or deterministic-session reconciliation.

## Decision Log

- Decision: Correct the unreleased fork API directly without aliases or shims.
  Rationale: the latest released base is `v0.18.3`; `pty_terminate` was added only on this fork
  after that tag.
  Date/Author: 2026-07-27 / Codex
- Decision: Keep Feather and Train types out of the SDK.
  Rationale: targeted process ownership is a sandbox-session primitive and must remain
  provider-neutral.
  Date/Author: 2026-07-27 / Codex
- Decision: Give every E2B launch a unique ownership token. Run non-TTY commands through a
  token-bearing Python supervisor in a dedicated POSIX session and discover owned process groups
  from that token during cleanup.
  Rationale: the supervisor keeps ownership visible when the user command scrubs its environment,
  while a UUID token avoids retrying against a raw PGID that may have been reused. TTY launches
  carry the same token so an ambiguous pre-handle create can be reconciled.
  Date/Author: 2026-07-27 / Codex
- Decision: Preserve Daytona's existing session deletion as the process-ownership boundary.
  Rationale: Daytona allocates a unique provider session before starting a non-TTY command, so
  deleting exactly that session is narrower and more reliable than adding POSIX process logic.
  Date/Author: 2026-07-27 / Codex
- Decision: Keep failed targeted and pruned cleanup entries in the provider registry with
  `termination_pending` set.
  Rationale: this blocks further writes while allowing later targeted or session-wide cleanup to
  retry the exact provider-owned process.
  Date/Author: 2026-07-27 / Codex
- Decision: Finish owned cleanup tasks before propagating caller cancellation.
  Rationale: shielding once is insufficient when a caller cancels repeatedly; the original
  cancellation must remain the outward result without abandoning provider cleanup.
  Date/Author: 2026-07-27 / Codex
- Decision: Serialize provider launch handoff against session-wide cleanup and retire capacity
  before starting replacement work.
  Rationale: a one-time registry snapshot cannot drain work that registers after the snapshot,
  and adding a replacement before pruning temporarily violates the advertised process cap.
  Date/Author: 2026-07-27 / Codex
- Decision: Bound ambiguous-create reconciliation rather than add unbounded provider retries.
  Rationale: E2B token discovery and Daytona's deterministic session ID can cover a short
  late-commit window without hiding persistent transport failure or blocking cancellation
  indefinitely.
  Date/Author: 2026-07-27 / Codex

## Outcomes & Retrospective

The fork now gives E2B commands token-based ownership and terminates non-TTY process groups with
bounded TERM/KILL escalation. Daytona retains its narrower deterministic provider-session
lifecycle. Both adapters serialize launch handoff against global cleanup, keep failed cleanup
registered for retry, preserve final output, and bound provider-launch cancellation independently
from command lifetime.

Focused E2B, Daytona, and base-session tests complete with 198 passing tests and two Linux-only
process tests skipped on macOS. The repository verification script completed formatting, Ruff,
mypy, pyright, and the full test suite successfully. Live Feather provider smokes remain
outstanding.

## Context and Orientation

The working tree is `/Users/daanishkhazi/Software/openai-agents-python-pty-terminate-v0183`,
based on release tag `v0.18.3`. The fork already adds `pty_terminate` to
`src/agents/sandbox/session/base_sandbox_session.py`, forwards it through
`src/agents/sandbox/session/sandbox_session.py`, and implements it in the E2B and Daytona
extensions. Feather pins this fork by immutable commit.

`pty_exec_start` starts provider-owned work, registers a random SDK process ID, and performs an
initial output collection before returning that ID. A caller cannot target cleanup until the ID is
returned. Therefore cancellation after registration but before return must be handled inside the
SDK.

## Plan of Work

First, settle how E2B commands acquire a stable ownership identity that can be terminated without
importing Feather policy into the SDK. Preserve Daytona's deterministic provider session deletion.
Serialize launch handoff against global cleanup, retire capacity before replacement launch, and
put post-registration output collection inside a cancellation boundary that terminates exactly
the owned entry before re-raising cancellation. Reuse the existing `_terminate_pty_entry` helpers
so targeted termination, pruning, cancellation, and global cleanup do not diverge.

Add focused mocked-provider tests beside the existing E2B and Daytona sandbox tests. The tests
must prove that cancellation after registration performs targeted cleanup, that terminal output
is drained, and that unrelated registered entries remain intact. Add a descendant test at the
lowest layer that can truthfully establish process-group behavior; do not encode Feather command
parsing in the SDK.

## Concrete Steps

Work from the SDK worktree:

    cd /Users/daanishkhazi/Software/openai-agents-python-pty-terminate-v0183
    uv run pytest <focused E2B and Daytona tests>
    bash .agents/skills/code-change-verification/scripts/run.sh

After verification, commit the SDK changes and update Feather and Train to one immutable SDK SHA.

## Validation and Acceptance

Acceptance requires:

- cancellation after provider registration but before `pty_exec_start` returns leaves no
  registered or remote process;
- stopping one process does not stop another;
- E2B non-TTY descendants in the managed process group cannot survive explicit stop, timeout, or
  caller cancellation;
- Daytona retains equivalent session-level cleanup;
- global cleanup cannot pass an in-flight launch, and concurrent replacement starts do not exceed
  the process cap;
- ambiguous E2B/Daytona creates are reconciled across a bounded late-commit window;
- final output remains available in `PtyExecUpdate`;
- the repository's format, lint, type-check, and test commands pass.

Live E2B and Daytona descendant/deadline checks will run from Feather's smoke package after the
fork is pinned there.

## Idempotence and Recovery

Tests and formatting commands are rerunnable. Provider cleanup tests use fakes and must leave no
external resources. Live sandbox checks are owned by Feather smoke fixtures, which close
caller-owned sessions even on failure. Dependency pins will change only after the SDK commit is
verified.

## Artifacts and Notes

E2B envd source at commit `8e89a410fd054006d6a506e2e7862e8602bbe94a` implements:

    return p.cmd.Process.Signal(signal)

That establishes PID-only signaling in the provider server and motivates explicit descendant
ownership.

## Interfaces and Dependencies

The public interface remains:

    await session.pty_terminate(session_id: int) -> PtyExecUpdate

No new Feather, Train, provider-selection, retry, or model-loop dependency may enter the SDK.
