import argparse
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception as error:
        raise RuntimeError(
            "Add a Kaggle secret named GITHUB_TOKEN with repository write access"
        ) from error
    if not token:
        raise RuntimeError(
            "Add a Kaggle secret named GITHUB_TOKEN with repository write access"
        )
    return token


def push_checkpoint(checkpoint_path: Path, branch: str, message: str) -> None:
    token = get_github_token()
    run(["git", "lfs", "install"])
    run(["git", "lfs", "track", "kaggle/output/*.pt"])
    run(["git", "config", "user.name", os.environ.get("GIT_USER_NAME", "kaggle-trainer")])
    run(
        [
            "git",
            "config",
            "user.email",
            os.environ.get("GIT_USER_EMAIL", "kaggle-trainer@users.noreply.github.com"),
        ]
    )
    run(["git", "add", ".gitattributes", str(checkpoint_path)])
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        return
    run(["git", "commit", "-m", message])

    with tempfile.TemporaryDirectory() as temporary_directory:
        askpass_path = Path(temporary_directory) / "git-askpass.sh"
        askpass_path.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "*Username*) printf '%s\\n' 'x-access-token' ;;\n"
            "*) printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass_path.chmod(askpass_path.stat().st_mode | stat.S_IXUSR)
        environment = os.environ.copy()
        environment["GITHUB_TOKEN"] = token
        environment["GIT_ASKPASS"] = str(askpass_path)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        run(["git", "push", "origin", f"HEAD:{branch}"], env=environment)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit-message", default="Add trained Jev checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        default="kaggle/output/best_model.pt",
    )
    return parser.parse_known_args()


def main() -> None:
    args, training_args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "jev_dev.train",
            "--checkpoint-path",
            str(checkpoint_path),
            *training_args,
        ]
    )
    push_checkpoint(checkpoint_path, args.branch, args.commit_message)


if __name__ == "__main__":
    main()
