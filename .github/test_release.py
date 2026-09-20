"""Release guards and archive contracts; no network, real tags, or GitHub writes."""

import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import release


ARM64_HEADER = b"\xcf\xfa\xed\xfe" + (0x0100000C).to_bytes(4, "little")
MACOS_LOAD_COMMANDS = """urb:
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 13.0
      sdk 15.5
   ntools 1
     tool 3
  version 1115.7.3
Load command 10
      cmd LC_SOURCE_VERSION
  cmdsize 16
  version 2026.9.21
Load command 11
          cmd LC_LOAD_DYLIB
      cmdsize 56
         name /usr/lib/libSystem.B.dylib (offset 24)
   time stamp 2 Thu Jan  1 00:00:02 1970
      current version 1351.0.0
compatibility version 1.0.0
"""


class ReleaseTests(unittest.TestCase):
    def test_release_tag_and_cargo_version_must_match(self):
        metadata = json.dumps({"packages": [{
            "manifest_path": str(release.ROOT / "Cargo.toml"), "version": "0.1.0",
        }]})
        with patch.object(release, "run", return_value=metadata):
            with self.assertRaisesRegex(ValueError, "does not match"):
                release.validate("v0.2.0")

    def test_reject_invalid_or_unsupported_tags(self):
        for tag in ("0.2.0", "v0.2", "v00.2.0", "v0.2.0-rc.1", "v0.2.0\nextra=x", "../../v0.2.0"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.tag_version(tag)

    def test_notes_select_only_the_requested_release(self):
        changelog = "# Changelog\n\n## [0.3.0]\n\nNext\n\n## [0.2.0] - 2026-09-21\n\n### Added\n\nTUI\n\n## [0.1.0]\n\nOld\n\n[0.1.0]: https://example.test\n"
        self.assertEqual(release.release_notes(changelog, "0.2.0"), "### Added\n\nTUI\n")
        self.assertEqual(release.release_notes(changelog, "0.1.0"), "Old\n")

    def test_notes_reject_missing_empty_and_duplicate_entries(self):
        for changelog in ("## [0.1.0]\nOld\n", "## [0.2.0]\n\n", "## [0.2.0]\nA\n## [0.2.0]\nB\n"):
            with self.subTest(changelog=changelog), self.assertRaises(ValueError):
                release.release_notes(changelog, "0.2.0")

    def test_linux_architecture_and_glibc_baseline(self):
        header = b"\x7fELF\x02\x01" + bytes(12) + (62).to_bytes(2, "little")
        release.check_runtime("x86_64-unknown-linux-gnu", header, "GLIBC_2.2.5 GLIBC_2.34 GLIBC_2.35")
        for invalid in ("GLIBC_2.36", "GLIBC_2.9 GLIBC_2.100", "no version information"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                release.check_runtime("x86_64-unknown-linux-gnu", header, invalid)
        with self.assertRaisesRegex(ValueError, "x86_64"):
            release.check_runtime("x86_64-unknown-linux-gnu", header[:18] + (183).to_bytes(2, "little"), "GLIBC_2.35")

    def test_mac_architecture_and_deployment_target(self):
        modern = "  cmd LC_BUILD_VERSION\n  platform 1\n  minos 13.0\n  sdk 15.5\n"
        legacy = "  cmd LC_VERSION_MIN_MACOSX\n  version 11.0\n"
        release.check_runtime("aarch64-apple-darwin", ARM64_HEADER, modern)
        release.check_runtime("aarch64-apple-darwin", ARM64_HEADER, legacy)
        for version in ("13.0.1", "13.1", "14.0"):
            for metadata in (modern.replace("minos 13.0", f"minos {version}"),
                             legacy.replace("version 11.0", f"version {version}")):
                with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "exceeds 13.0"):
                    release.check_runtime("aarch64-apple-darwin", ARM64_HEADER, metadata)
        with self.assertRaisesRegex(ValueError, "arm64"):
            release.check_runtime("aarch64-apple-darwin", b"\xcf\xfa\xed\xfe" + (0x01000007).to_bytes(4, "little"), "minos 13.0\n")

    def test_mac_ignores_linker_sdk_source_and_dylib_versions(self):
        for platform in ("1", "MACOS", "macos"):
            with self.subTest(platform=platform):
                details = MACOS_LOAD_COMMANDS.replace("platform 1", f"platform {platform}")
                release.check_runtime("aarch64-apple-darwin", ARM64_HEADER, details)

    def test_mac_legacy_minimum_is_scoped_to_its_load_command(self):
        details = """Load command 1
      cmd LC_VERSION_MIN_MACOSX
  cmdsize 16
  version 11.0
      sdk 15.5
Load command 2
      cmd LC_SOURCE_VERSION
  cmdsize 16
  version 2026.9.21
"""
        release.check_runtime("aarch64-apple-darwin", ARM64_HEADER, details)

    def test_mac_rejects_missing_malformed_or_non_macos_deployment_records(self):
        for details in (
            "minos 13.0\n",  # a value outside a load command is not deployment metadata
            "cmd LC_SOURCE_VERSION\nversion 11.0\n",
            MACOS_LOAD_COMMANDS.replace("    minos 13.0\n", ""),
            MACOS_LOAD_COMMANDS.replace("minos 13.0", "minos invalid"),
            MACOS_LOAD_COMMANDS.replace("minos 13.0", "minos 13.0\nminos 14.0"),
            MACOS_LOAD_COMMANDS.replace(" platform 1\n", ""),
            MACOS_LOAD_COMMANDS.replace("platform 1", "platform 2"),
            MACOS_LOAD_COMMANDS.replace("platform 1", "platform IOS"),
        ):
            with self.subTest(details=details), self.assertRaises(ValueError):
                release.check_runtime("aarch64-apple-darwin", ARM64_HEADER, details)

    def test_mac_checks_every_deployment_record(self):
        details = MACOS_LOAD_COMMANDS + "Load command 12\ncmd LC_VERSION_MIN_MACOSX\nversion 14.0\n"
        with self.assertRaisesRegex(ValueError, "deployment target 14.0 exceeds 13.0"):
            release.check_runtime("aarch64-apple-darwin", ARM64_HEADER, details)

    def test_package_reports_runtime_metadata_on_baseline_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            binary = output / "urb"
            binary.write_bytes(ARM64_HEADER)
            details = MACOS_LOAD_COMMANDS.replace("minos 13.0", "minos 14.0")
            with patch.object(release, "validate", return_value="0.2.0"), \
                    patch.object(release.platform, "system", return_value="Darwin"), \
                    patch.object(release.platform, "machine", return_value="arm64"), \
                    patch.object(release, "run", return_value=details):
                with self.assertRaises(ValueError) as failure:
                    release.package_release("v0.2.0", "aarch64-apple-darwin", binary, output)
            self.assertIn("deployment target 14.0 exceeds 13.0", str(failure.exception))
            self.assertIn(details, str(failure.exception))

    def test_package_rejects_the_wrong_host(self):
        with patch.object(release, "validate", return_value="0.2.0"), \
                patch.object(release.platform, "system", return_value="Linux"), \
                patch.object(release.platform, "machine", return_value="x86_64"):
            with self.assertRaisesRegex(ValueError, "native"):
                release.package_release("v0.2.0", "aarch64-apple-darwin", Path("unused"), Path("unused"))

    def test_archives_include_executable_docs_and_build_info(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            binary = output / "urb"
            binary.write_bytes(b"test binary")
            binary.chmod(0o644)  # Packaging sets executable permissions explicitly.
            info = {"version": "0.2.0", "source_dirty": True}
            archive = release.make_archive(binary, output, "v0.2.0", "x86_64-unknown-linux-gnu", info)
            base = archive.name.removesuffix(".tar.gz")
            with tarfile.open(archive, "r:gz") as bundle:
                self.assertEqual(bundle.getmember(f"{base}/urb").mode & 0o777, 0o755)
                self.assertEqual(bundle.extractfile(f"{base}/urb").read(), b"test binary")
                self.assertEqual(json.load(bundle.extractfile(f"{base}/BUILD_INFO.json")), info)
                for name in (*release.DOCUMENTS, "INSTALL.md"):
                    self.assertTrue(bundle.getmember(f"{base}/{name}").isfile(), name)
            self.assertEqual(
                (output / (archive.name + ".sha256")).read_text(),
                f"{release.digest(archive)}  {archive.name}\n",
            )

    def write_assets(self, output):
        for target in release.TARGETS:
            name = release.archive_name("v0.2.0", target)
            path = output / name
            path.write_bytes(target.encode())
            (output / (name + ".sha256")).write_text(f"{release.digest(path)}  {name}\n")

    def test_collect_requires_both_archives_and_rejects_extra_ones(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(ValueError, "exactly"):
                release.collect("v0.2.0", output)
            self.write_assets(output)
            (output / "unexpected.tar.gz").write_bytes(b"other version or platform")
            with self.assertRaisesRegex(ValueError, "exactly"):
                release.collect("v0.2.0", output)
            self.assertFalse((output / "SHA256SUMS").exists())

    def test_collect_rejects_modified_archive_and_missing_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_assets(output)
            name = release.archive_name("v0.2.0", "aarch64-apple-darwin")
            (output / name).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                release.collect("v0.2.0", output)
            (output / (name + ".sha256")).unlink()
            with self.assertRaises(FileNotFoundError):
                release.collect("v0.2.0", output)
            self.assertFalse((output / "SHA256SUMS").exists())
            self.assertFalse((output / "RELEASE_NOTES.md").exists())

    def test_collect_writes_combined_checksums_and_current_notes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_assets(output)
            release.collect("v0.2.0", output)
            lines = (output / "SHA256SUMS").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                expected, name = line.split("  ")
                self.assertEqual(release.digest(output / name), expected)
            notes = (output / "RELEASE_NOTES.md").read_text()
            self.assertIn("/inspect", notes)
            self.assertNotIn("Initial public release", notes)


if __name__ == "__main__":
    unittest.main()
