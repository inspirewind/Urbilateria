"""Prepare native release archives using only Python's standard library (3.9+)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "x86_64-unknown-linux-gnu": ("Linux", "x86_64", "Linux x86_64, glibc >= 2.35"),
    "aarch64-apple-darwin": ("Darwin", "arm64", "Apple Silicon, macOS >= 13.0"),
}
DOCUMENTS = (
    "LICENSE", "README.md", "README_CN.md", "CHANGELOG.md", "RELEASING.md",
    "assets/urbilateria-mark.svg",
)


def run(*args):
    return subprocess.run(
        args, cwd=ROOT, check=True, capture_output=True, text=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout.strip()


def tag_version(tag):
    if not re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", tag):
        raise ValueError("Release tags must use vMAJOR.MINOR.PATCH (stable releases only)")
    return tag[1:]


def release_notes(changelog, version):
    sections = re.split(r"^## ", changelog, flags=re.MULTILINE)
    matches = []
    for section in sections[1:]:
        heading, _, body = section.partition("\n")
        if re.fullmatch(r"\[" + re.escape(version) + r"\](?: - \d{4}-\d{2}-\d{2})?", heading):
            # Link definitions at the end of the changelog are not release notes.
            body = re.split(r"^\[[^\]]+\]:\s", body, maxsplit=1, flags=re.MULTILINE)[0]
            matches.append(body.strip())
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"CHANGELOG.md must have one nonempty ## [{version}] section")
    return matches[0] + "\n"


def validate(tag):
    version = tag_version(tag)
    metadata = json.loads(run("cargo", "+1.88.0", "metadata", "--locked", "--no-deps", "--format-version", "1"))
    packages = [package for package in metadata["packages"]
                if Path(package["manifest_path"]).resolve() == ROOT / "Cargo.toml"]
    if len(packages) != 1 or packages[0]["version"] != version:
        raise ValueError(f"Tag {tag} does not match the root Cargo.toml version")
    release_notes((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    return version


def version_tuple(text):
    parts = tuple(int(part) for part in text.split("."))
    return parts + (0,) * (3 - len(parts))


def macos_deployment_versions(details):
    # "version" also appears in build-tool and source-version records. Interpret it
    # only inside LC_VERSION_MIN_MACOSX; LC_BUILD_VERSION uses "minos" for the OS.
    commands = re.split(r"^[ \t]*cmd[ \t]+(\S+)[ \t]*$", details, flags=re.MULTILINE)
    versions = []
    for command, body in zip(commands[1::2], commands[2::2]):
        if command == "LC_BUILD_VERSION":
            platforms = re.findall(r"^[ \t]*platform[ \t]+(\S+)[ \t]*$", body, re.MULTILINE)
            if len(platforms) != 1 or platforms[0].upper() not in ("1", "MACOS"):
                raise ValueError(f"Expected macOS platform in LC_BUILD_VERSION; found {platforms!r}")
            field = "minos"
        elif command == "LC_VERSION_MIN_MACOSX":
            field = "version"
        else:
            continue
        values = re.findall(r"^[ \t]*" + field + r"[ \t]+(\d+(?:\.\d+){0,2})[ \t]*$",
                            body, re.MULTILINE)
        if len(values) != 1:
            raise ValueError(f"Expected one valid {field} in {command}; found {values!r}")
        versions.extend(values)
    if not versions:
        raise ValueError("No macOS deployment target found in otool load commands")
    return versions


def check_runtime(target, header, details):
    """Check binary architecture and the OS version recorded by the native linker."""
    if target == "x86_64-unknown-linux-gnu":
        if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 62:
            raise ValueError("Expected a 64-bit x86_64 Linux ELF binary")
        versions = re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)\b", details)
        if not versions or max(map(version_tuple, versions)) > (2, 35, 0):
            raise ValueError("Linux binary must use glibc 2.35 or older symbols; build on Ubuntu 22.04")
    else:
        if header[:4] != b"\xcf\xfa\xed\xfe" or int.from_bytes(header[4:8], "little") != 0x0100000C:
            raise ValueError("Expected an Apple Silicon arm64 Mach-O binary")
        minimum = max(macos_deployment_versions(details), key=version_tuple)
        if version_tuple(minimum) > (13, 0, 0):
            raise ValueError(f"macOS deployment target {minimum} exceeds 13.0; set MACOSX_DEPLOYMENT_TARGET=13.0")


def archive_name(tag, target):
    return f"urb-{tag}-{target}.tar.gz"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_archive(binary, output, tag, target, build_info):
    output.mkdir(parents=True, exist_ok=True)
    name = archive_name(tag, target)
    archive = output / name
    with tempfile.TemporaryDirectory(prefix="urb-release-") as temporary:
        package = Path(temporary) / name.removesuffix(".tar.gz")
        package.mkdir()
        shutil.copyfile(binary, package / "urb")
        (package / "urb").chmod(0o755)
        for document in DOCUMENTS:
            destination = package / document
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / document, destination)
        (package / "BUILD_INFO.json").write_text(json.dumps(build_info, indent=2) + "\n", encoding="utf-8")
        (package / "INSTALL.md").write_text(
            f"# Urbilateria {tag_version(tag)}\n\n"
            f"Platform: {TARGETS[target][2]}.\n\n"
            "Before extracting, verify the downloaded archive against SHA256SUMS:\n"
            "`sha256sum --ignore-missing -c SHA256SUMS` on Linux, or\n"
            "`shasum -a 256 -c SHA256SUMS` on macOS (the other platform's absent archive\n"
            "may be reported as missing; your downloaded archive must report OK).\n\n"
            "Run `./urb --version`, `./urb help`, or `./urb ui` from this directory.\n"
            "Optionally copy `urb` to a directory on your PATH. No Rust, Python, Node.js,\n"
            "or model weights are required for startup. Model commands need your own\n"
            "checkpoint directory. In the TUI, start with `/inspect /path/to/model`.\n\n"
            "On macOS, pass `--ram-gib` explicitly to `/plan`. macOS archives are\n"
            "not Developer ID signed or notarized. If Gatekeeper blocks execution,\n"
            "use the system's Privacy & Security controls to review the downloaded app.\n\n"
            "See README.md or README_CN.md for commands and model-specific limits.\n",
            encoding="utf-8",
        )
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
            bundle.add(package, arcname=package.name)
    (output / (name + ".sha256")).write_text(f"{digest(archive)}  {name}\n", encoding="utf-8")
    return archive


def package_release(tag, target, binary, output):
    version = validate(tag)
    system, machine, baseline = TARGETS[target]
    if (platform.system(), platform.machine()) != (system, machine):
        raise ValueError(f"Package {target} on its native {system}/{machine} host")
    binary = binary.resolve()
    with binary.open("rb") as source:
        header = source.read(32)
    details = (run("readelf", "--version-info", str(binary)) if system == "Linux"
               else run("otool", "-l", str(binary)))
    try:
        check_runtime(target, header, details)
    except ValueError as error:
        raise ValueError(f"{error}\nInspected runtime metadata for {target}:\n{details}") from error
    if run(str(binary), "--version") != f"urb {version}":
        raise ValueError("Binary version does not match the tag; rebuild before packaging")
    run(str(binary), "help")
    run(str(binary), "ui", "--help")
    build_info = {
        "version": version,
        "target": target,
        "source_commit": run("git", "rev-parse", "HEAD"),
        "source_dirty": bool(run("git", "status", "--porcelain")),
        "rustc": run("rustc", "+1.88.0", "--version"),
        "runtime_baseline": baseline,
    }
    return make_archive(binary, output, tag, target, build_info)


def collect(tag, output):
    version = tag_version(tag)
    notes = release_notes((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    expected = sorted(archive_name(tag, target) for target in TARGETS)
    if sorted(path.name for path in output.glob("*.tar.gz")) != expected:
        raise ValueError("Expected exactly the Linux x86_64 and Apple Silicon archives for this tag")
    sums = []
    for name in expected:
        checksum = output / (name + ".sha256")
        line = f"{digest(output / name)}  {name}\n"
        if checksum.read_text(encoding="utf-8") != line:
            raise ValueError(f"Checksum mismatch for {name}")
        sums.append(line)
    # Write final files only after both artifacts have passed verification.
    (output / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    (output / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate")
    validation.add_argument("--tag", required=True)
    validation.add_argument("--github-output", type=Path)
    package = commands.add_parser("package")
    package.add_argument("--tag", required=True)
    package.add_argument("--target", choices=TARGETS, required=True)
    package.add_argument("--binary", type=Path, help="default: target/TARGET/release/urb")
    package.add_argument("--output", type=Path, default=ROOT / "target/release-assets")
    collection = commands.add_parser("collect")
    collection.add_argument("--tag", required=True)
    collection.add_argument("--output", type=Path, default=ROOT / "target/release-assets")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            version = validate(args.tag)
            if args.github_output:
                with args.github_output.open("a", encoding="utf-8") as output:
                    output.write(f"version={version}\ncommit={run('git', 'rev-parse', 'HEAD')}\n")
            print(f"Validated {args.tag}")
        elif args.command == "package":
            binary = args.binary or ROOT / "target" / args.target / "release/urb"
            print(package_release(args.tag, args.target, binary, args.output))
        else:
            collect(args.tag, args.output)
            print(f"Verified both archives; wrote SHA256SUMS and RELEASE_NOTES.md in {args.output}")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        parser.exit(1, f"Release preparation failed: {detail}\n")


if __name__ == "__main__":
    main()
