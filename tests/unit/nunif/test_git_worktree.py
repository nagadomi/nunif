import os

from tests.helpers.git_worktree import git_worktree, is_git_ref, resolve_repo_path


def test_is_git_ref() -> None:
    assert is_git_ref("HEAD")
    assert is_git_ref("HEAD~0")
    assert not is_git_ref("non_existent_commit_hash_123456789")


def test_git_worktree_checkout_and_cleanup() -> None:
    worktree_path_saved = None
    with git_worktree("HEAD") as worktree_path:
        worktree_path_saved = worktree_path
        assert os.path.exists(worktree_path)
        # Check that files in repository exist in worktree
        assert os.path.exists(os.path.join(worktree_path, "requirements.txt"))

    # Verify that worktree path was removed
    assert not os.path.exists(worktree_path_saved)


def test_resolve_repo_path_directory() -> None:
    # Existing directory should be resolved directly
    with resolve_repo_path(".") as path:
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, "requirements.txt"))
