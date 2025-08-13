#!/usr/bin/env python3
"""Update version in pyproject.toml based on git repository information."""

import subprocess
import re
import sys
from pathlib import Path


def get_latest_tag():
    """Get the latest git tag that starts with 'v'."""
    try:
        result = subprocess.run(
            ["git", "tag"],
            capture_output=True,
            text=True,
            check=True
        )
        tags = [tag for tag in result.stdout.splitlines() if tag.startswith("v")]
        if not tags:
            return None
        # Sort version tags properly
        tags.sort(key=lambda x: [int(n) for n in x[1:].split(".")])
        return tags[-1]
    except subprocess.CalledProcessError:
        return None


def get_commits_since_tag(tag):
    """Get number of commits since the given tag."""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", f"{tag}..HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError:
        return 0


def get_commit_hash():
    """Get the first 8 characters of the current commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "00000000"


def update_pyproject_version():
    """Update the version in pyproject.toml."""
    # Get git information
    latest_tag = get_latest_tag()
    if not latest_tag:
        print("No git tag found starting with 'v'", file=sys.stderr)
        return False
    
    # Remove 'v' prefix from tag
    base_version = latest_tag[1:]
    distance = get_commits_since_tag(latest_tag)
    checksum = get_commit_hash()
    
    # Format version string
    if distance > 0:
        version = f"{base_version}.post{distance}+{checksum}"
    else:
        version = f"{base_version}"
        
    
    # Read pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("pyproject.toml not found", file=sys.stderr)
        return False
    
    content = pyproject_path.read_text()
    
    # Update version line
    updated_content = re.sub(
        r'^version = ".*"',
        f'version = "{version}"',
        content,
        flags=re.MULTILINE
    )
    
    # Write back to file
    pyproject_path.write_text(updated_content)
    
    print(f"Updated version to: {version}")
    return True


if __name__ == "__main__":
    success = update_pyproject_version()
    sys.exit(0 if success else 1)
