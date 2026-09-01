from __future__ import annotations

from typing import Final

STAT_SCRIPT: Final[str] = """
import grp
import json
import os
import pwd
import stat
import sys

root, target, *grant_roots = sys.argv[1:]


def emit(payload):
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


lexical_roots = [root, *grant_roots]
containing_roots = []
for allowed_root in lexical_roots:
    try:
        if os.path.commonpath((allowed_root, target)) == allowed_root:
            containing_roots.append(allowed_root)
    except ValueError:
        pass
if not containing_roots:
    emit({"resolved_path": target, "status": "escape"})
    raise SystemExit(0)
boundary = max(containing_roots, key=lambda path: len(path.split(os.sep)))

resolved_root = os.path.realpath(root)
resolved_grant_roots = []
for grant_root in grant_roots:
    resolved_grant = os.path.realpath(grant_root)
    if resolved_grant == os.sep:
        emit({"component": grant_root, "status": "invalid_grant_root"})
        raise SystemExit(0)
    resolved_grant_roots.append(resolved_grant)

resolved_target = os.path.realpath(target)
resolved_allowed_roots = [resolved_root, *resolved_grant_roots]
try:
    target_is_allowed = any(
        os.path.commonpath((allowed_root, resolved_target)) == allowed_root
        for allowed_root in resolved_allowed_roots
    )
except ValueError:
    target_is_allowed = False
if not target_is_allowed:
    emit({"resolved_path": resolved_target, "status": "escape"})
    raise SystemExit(0)

try:
    boundary_metadata = os.stat(boundary)
except FileNotFoundError:
    status = "missing" if boundary == target else "missing_ancestor"
    emit({"component": boundary, "status": status})
    raise SystemExit(0)
except NotADirectoryError:
    emit({"component": os.path.dirname(boundary), "status": "not_directory"})
    raise SystemExit(0)

if boundary == target:
    metadata = boundary_metadata
else:
    if not stat.S_ISDIR(boundary_metadata.st_mode):
        emit({"component": boundary, "status": "not_directory"})
        raise SystemExit(0)

    relative = os.path.relpath(target, boundary)
    parts = relative.split(os.sep)
    current = boundary
    for part in parts[:-1]:
        current = os.path.join(current, part)
        try:
            component_metadata = os.stat(current)
        except FileNotFoundError:
            emit({"component": current, "status": "missing_ancestor"})
            raise SystemExit(0)
        except NotADirectoryError:
            emit({"component": os.path.dirname(current), "status": "not_directory"})
            raise SystemExit(0)
        if not stat.S_ISDIR(component_metadata.st_mode):
            emit({"component": current, "status": "not_directory"})
            raise SystemExit(0)

    try:
        metadata = os.stat(target)
    except FileNotFoundError:
        emit({"component": target, "status": "missing"})
        raise SystemExit(0)
    except NotADirectoryError:
        emit({"component": os.path.dirname(target), "status": "not_directory"})
        raise SystemExit(0)

try:
    owner = pwd.getpwuid(metadata.st_uid).pw_name
except KeyError:
    owner = str(metadata.st_uid)
try:
    group = grp.getgrgid(metadata.st_gid).gr_name
except KeyError:
    group = str(metadata.st_gid)

emit(
    {
        "group": group,
        "mode": metadata.st_mode,
        "owner": owner,
        "size": metadata.st_size,
        "status": "entry",
    }
)
""".strip()
