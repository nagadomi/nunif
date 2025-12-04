import os


def get_home_dir(project_name, default_path=None):
    nunif_home = os.getenv("NUNIF_HOME")
    if nunif_home:
        nunif_home = os.path.expanduser(nunif_home)
        if project_name:
            return os.path.normpath(os.path.join(nunif_home, project_name))
        else:
            return os.path.normpath(nunif_home)

    if default_path is None:
        repository_root = os.path.join(os.path.dirname(__file__), "..", "..")
        if project_name:
            return os.path.normpath(os.path.join(repository_root, project_name))
        else:
            return os.path.normpath(repository_root)

    return os.path.normpath(default_path)


def ensure_home_dir(project_name, default_path=None):
    home_dir = get_home_dir(project_name, default_path)
    os.makedirs(home_dir, exist_ok=True)
    return home_dir


def is_nunif_home_set():
    return bool(os.getenv("NUNIF_HOME"))
