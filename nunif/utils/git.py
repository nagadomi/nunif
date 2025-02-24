from os import path


def get_current_branch(root=None):
    if root is None:
        head_path = path.join(path.dirname(__file__), "..", "..", ".git", "HEAD")
    else:
        head_path = path.join(root, ".git", "HEAD")

    if path.exists(head_path):
        with open(head_path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line[0:5] == "ref: ":
                    branch_path = line[5:]
                    branch_name = branch_path.split("/")[-1]
                    return branch_name
    return None


def _test_get_current_branch():
    print(get_current_branch())
    print(get_current_branch(root="dummy_path"))


if __name__ == "__main__":
    _test_get_current_branch()
