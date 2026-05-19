import pathlib
import subprocess
import sys


def run_git(cmd):
    """Helper to run git commands and exit on failure."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Git Error: {result.stderr}")
        sys.exit(1)


def main():
    path = pathlib.Path("pyproject.toml")
    if not path.exists():
        print("pyproject.toml not found!")
        sys.exit(1)

    # 1. Edit the file
    content = path.read_text()
    lines = content.splitlines()
    new_v = ""
    for i, line in enumerate(lines):
        if line.strip().startswith("version ="):
            old_v = line.split('"')[1]
            major, minor, patch = map(int, old_v.split("."))
            new_v = f"{major}.{minor}.{patch + 1}"
            lines[i] = f'version = "{new_v}"'
            break

    path.write_text("\n".join(lines) + "\n")
    print(f"Local file updated to {new_v}")

    # 2. Git Commands
    # We configure the 'bot' identity so GitHub knows who made the change
    run_git('git config user.name "github-actions[bot]"')
    run_git('git config user.email "github-actions[bot]@users.noreply.github.com"')

    run_git("git add pyproject.toml")

    # [skip ci] is vital to prevent an infinite loop of actions!
    run_git(f'git commit -m "chore: auto-bump version to {new_v} [skip ci]"')

    run_git("git push")
    print(f"Successfully pushed version {new_v} to the server.")


if __name__ == "__main__":
    main()
