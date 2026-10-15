import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def is_git_ref(ref: str, cwd: str = PROJECT_DIR) -> bool:
    res = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
    )
    return res.returncode == 0


@contextmanager
def git_worktree(commit_ref: str, cwd: str = PROJECT_DIR) -> Generator[str, None, None]:
    if not is_git_ref(commit_ref, cwd=cwd):
        raise ValueError(f"Invalid git reference or commit hash: {commit_ref}")

    temp_dir = tempfile.mkdtemp(prefix=f"nunif_worktree_{commit_ref.replace('/', '_')}_")
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", temp_dir, commit_ref],
            cwd=cwd,
            check=True,
            capture_output=True,
        )
        yield os.path.abspath(temp_dir)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", temp_dir],
            cwd=cwd,
            check=False,
            capture_output=True,
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=cwd,
            check=False,
            capture_output=True,
        )


@contextmanager
def resolve_repo_path(repo_or_ref: str, cwd: str = PROJECT_DIR) -> Generator[str, None, None]:
    if os.path.isdir(repo_or_ref):
        yield os.path.abspath(repo_or_ref)
    else:
        with git_worktree(repo_or_ref, cwd=cwd) as worktree_path:
            yield worktree_path
