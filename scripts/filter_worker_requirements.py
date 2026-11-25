"""Emit plagiarism-only locked packages, reusing versions from the dropout lock."""
import re
import sys
from pathlib import Path


def package_blocks(path: Path):
    blocks = []
    current = []
    for line in path.read_text().splitlines():
        if re.match(r"^[A-Za-z0-9_.-]+(?:\[[^]]+\])?==", line):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def package_name(block):
    return re.match(r"^([A-Za-z0-9_.-]+)", block[0]).group(1).lower().replace("_", "-")


def main():
    primary, plagiarism, destination = map(Path, sys.argv[1:4])
    shared = {package_name(block) for block in package_blocks(primary)}
    blocks = [block for block in package_blocks(plagiarism) if package_name(block) not in shared]
    # Index directives are omitted; the base Python image uses PyPI, while the
    # torch CPU wheel URL is expressed as an explicit find-links index below.
    with destination.open("w") as stream:
        stream.write("--index-url https://pypi.org/simple\n--extra-index-url https://download.pytorch.org/whl/cpu\n")
        stream.write("\n".join("\n".join(block) for block in blocks) + "\n")


if __name__ == "__main__":
    main()
