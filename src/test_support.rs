//! Filesystem fixtures shared by unit and integration tests; excluded from production builds.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir_with_nonce(prefix, nonce)
}

pub(crate) fn temp_dir_with_nonce(prefix: &str, nonce: u128) -> PathBuf {
    // Timestamps can repeat across threads, especially on clocks with coarse resolution.
    // Claim the directory atomically so tests never share or delete each other's fixtures.
    let mut attempt = 0_u64;
    loop {
        let path =
            std::env::temp_dir().join(format!("{prefix}_{}_{nonce}_{attempt}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return path,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => attempt += 1,
            Err(error) => panic!("cannot create fixture {}: {error}", path.display()),
        }
    }
}
