//! A bounded, header-only safetensors index with explicit random-access reads.
//!
//! Large checkpoints must not be memory-mapped wholesale: this reader opens each shard once,
//! reads at most a bounded JSON header, and retains only its handle and tensor offsets. Tensor
//! payloads are fetched with explicit positional ranges later.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
#[cfg(target_os = "linux")]
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
#[cfg(target_os = "linux")]
use std::fs::OpenOptions;
use std::fs::{self, File};
#[cfg(unix)]
use std::io;
use std::io::Read;
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::mem::{ManuallyDrop, MaybeUninit};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::ptr::NonNull;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SINGLE_READ_BYTES: usize = 64 * 1024 * 1024;

#[cfg(target_os = "linux")]
const DIRECT_ALIGNMENT: usize = 4 * 1024;
#[cfg(target_os = "linux")]
const O_DIRECT: i32 = 0o40000;

#[cfg(target_os = "linux")]
fn direct_expert_io_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("URB_DISABLE_DIRECT_EXPERT_IO").is_none())
}

/// Owned tensor bytes. Linux direct reads retain their page-aligned allocation so the bytes can
/// become matrix storage without a full-size copy and can be recycled after cache eviction.
#[derive(Debug)]
pub(crate) enum ReadBuffer {
    Heap(Vec<u8>),
    #[cfg(target_os = "linux")]
    Direct(DirectReadBuffer),
}

impl ReadBuffer {
    pub(crate) fn len(&self) -> usize {
        self.as_ref().len()
    }

    pub(crate) fn into_vec(self) -> Vec<u8> {
        match self {
            Self::Heap(bytes) => bytes,
            #[cfg(target_os = "linux")]
            Self::Direct(bytes) => bytes.as_ref().to_vec(),
        }
    }
}

impl From<Vec<u8>> for ReadBuffer {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Heap(bytes)
    }
}

impl AsRef<[u8]> for ReadBuffer {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Heap(bytes) => bytes,
            #[cfg(target_os = "linux")]
            Self::Direct(bytes) => bytes.as_ref(),
        }
    }
}

impl std::ops::Deref for ReadBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl Clone for ReadBuffer {
    fn clone(&self) -> Self {
        Self::Heap(self.as_ref().to_vec())
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
pub(crate) struct DirectReadBuffer {
    allocation: Option<NonNull<u8>>,
    capacity: usize,
    start: usize,
    len: usize,
}

#[cfg(target_os = "linux")]
impl DirectReadBuffer {
    fn with_capacity(length: usize) -> Option<Self> {
        let capacity = length
            .checked_add(DIRECT_ALIGNMENT - 1)
            .map(|value| value / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT)?;
        let layout = Layout::from_size_align(capacity, DIRECT_ALIGNMENT).ok()?;
        // SAFETY: `layout` is non-zero and has a supported power-of-two alignment.
        let allocation = NonNull::new(unsafe { alloc(layout) })?;
        Some(Self {
            allocation: Some(allocation),
            capacity,
            start: 0,
            len: 0,
        })
    }

    fn ensure_capacity(&mut self, length: usize) -> bool {
        if length <= self.capacity && self.allocation.is_some() {
            return true;
        }
        let Some(capacity) = length
            .checked_add(DIRECT_ALIGNMENT - 1)
            .map(|value| value / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT)
        else {
            return false;
        };
        let Ok(layout) = Layout::from_size_align(capacity, DIRECT_ALIGNMENT) else {
            return false;
        };
        // SAFETY: `layout` is non-zero and has a supported power-of-two alignment.
        let Some(allocation) = NonNull::new(unsafe { alloc(layout) }) else {
            return false;
        };
        if let Some(previous) = self.allocation.replace(allocation) {
            let previous_layout = Layout::from_size_align(self.capacity, DIRECT_ALIGNMENT)
                .expect("an existing direct-read allocation has valid layout");
            // SAFETY: `previous` was allocated with exactly `previous_layout` in this method.
            unsafe { dealloc(previous.as_ptr(), previous_layout) };
        }
        self.capacity = capacity;
        true
    }

    fn allocation_ptr(&mut self) -> *mut u8 {
        self.allocation
            .expect("a direct-read buffer is allocated before use")
            .as_ptr()
    }

    fn set_range(&mut self, start: usize, len: usize) {
        debug_assert!(start.saturating_add(len) <= self.capacity);
        self.start = start;
        self.len = len;
    }
}

#[cfg(target_os = "linux")]
impl AsRef<[u8]> for DirectReadBuffer {
    fn as_ref(&self) -> &[u8] {
        let allocation = self
            .allocation
            .expect("a direct-read buffer is allocated before use");
        // SAFETY: successful direct reads initialize `len` bytes beginning at `start`, and the
        // allocation remains uniquely owned for mutation or immutably borrowed for matrix use.
        unsafe { std::slice::from_raw_parts(allocation.as_ptr().add(self.start), self.len) }
    }
}

// The allocation owns plain bytes. It is mutated only while uniquely held by a loader and is
// immutable after being installed in a matrix, matching Vec<u8>'s Send + Sync guarantees.
#[cfg(target_os = "linux")]
unsafe impl Send for DirectReadBuffer {}
#[cfg(target_os = "linux")]
unsafe impl Sync for DirectReadBuffer {}

#[cfg(target_os = "linux")]
impl Drop for DirectReadBuffer {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation {
            let layout = Layout::from_size_align(self.capacity, DIRECT_ALIGNMENT)
                .expect("an existing direct-read allocation has valid layout");
            // SAFETY: `allocation` was allocated with exactly `layout` in `ensure_capacity`.
            unsafe { dealloc(allocation.as_ptr(), layout) };
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    /// Unsigned exponent-only E8M0 scale used by MXFP8/MXFP4 checkpoints.
    F8E8M0,
    Unknown(String),
}

impl DType {
    pub fn parse(value: &str) -> Self {
        match value {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3" | "F8_E4M3FN" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E8M0" | "F8_E8M0FNU" => Self::F8E8M0,
            other => Self::Unknown(other.to_owned()),
        }
    }

    pub fn element_bytes(&self) -> Option<u64> {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 => {
                Some(1)
            }
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::Unknown(_) => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => f.write_str("BOOL"),
            Self::U8 => f.write_str("U8"),
            Self::I8 => f.write_str("I8"),
            Self::U16 => f.write_str("U16"),
            Self::I16 => f.write_str("I16"),
            Self::U32 => f.write_str("U32"),
            Self::I32 => f.write_str("I32"),
            Self::U64 => f.write_str("U64"),
            Self::I64 => f.write_str("I64"),
            Self::F16 => f.write_str("F16"),
            Self::Bf16 => f.write_str("BF16"),
            Self::F32 => f.write_str("F32"),
            Self::F64 => f.write_str("F64"),
            Self::F8E4M3 => f.write_str("F8_E4M3"),
            Self::F8E5M2 => f.write_str("F8_E5M2"),
            Self::F8E8M0 => f.write_str("F8_E8M0"),
            Self::Unknown(value) => f.write_str(value),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensorDescriptor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Default)]
struct StrictStringMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for StrictStringMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StringMapVisitor;

        impl<'de> Visitor<'de> for StringMapVisitor {
            type Value = StrictStringMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object with unique string keys and values")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom(format!("duplicate metadata key {key:?}")));
                    }
                    values.insert(key, map.next_value::<String>()?);
                }
                Ok(StrictStringMap(values))
            }
        }

        deserializer.deserialize_map(StringMapVisitor)
    }
}

#[derive(Debug, Default)]
struct RawHeader {
    metadata: BTreeMap<String, String>,
    tensors: Vec<(String, RawTensorDescriptor)>,
}

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = RawHeader;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a safetensors header object with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut seen = HashSet::new();
                let mut metadata = None;
                let mut tensors = Vec::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !seen.insert(name.clone()) {
                        return Err(de::Error::custom(format!(
                            "duplicate top-level key {name:?}"
                        )));
                    }
                    if name == "__metadata__" {
                        metadata = Some(map.next_value::<StrictStringMap>()?.0);
                    } else {
                        tensors.push((name, map.next_value::<RawTensorDescriptor>()?));
                    }
                }
                Ok(RawHeader {
                    metadata: metadata.unwrap_or_default(),
                    tensors,
                })
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub shard: PathBuf,
    /// Absolute byte offset from the beginning of `shard`.
    pub data_offset: u64,
    pub data_len: u64,
    pub declared_elements: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShardInfo {
    pub path: PathBuf,
    pub file_len: u64,
    pub header_len: u64,
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug)]
pub enum SafetensorError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    NoShards(PathBuf),
    Invalid {
        path: PathBuf,
        reason: String,
    },
    DuplicateTensor {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },
    MissingTensor(String),
    ReadTooLarge {
        requested: usize,
        maximum: usize,
    },
    AllocationTooLarge {
        name: String,
        requested: u64,
        maximum: u64,
    },
}

impl fmt::Display for SafetensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::NoShards(path) => write!(f, "no .safetensors shards found in {}", path.display()),
            Self::Invalid { path, reason } => {
                write!(f, "invalid safetensors shard {}: {reason}", path.display())
            }
            Self::DuplicateTensor {
                name,
                first,
                second,
            } => write!(
                f,
                "tensor {name:?} occurs in both {} and {}",
                first.display(),
                second.display()
            ),
            Self::MissingTensor(name) => write!(f, "tensor {name:?} was not found"),
            Self::ReadTooLarge { requested, maximum } => write!(
                f,
                "refusing a {requested}-byte tensor read; per-read limit is {maximum} bytes"
            ),
            Self::AllocationTooLarge {
                name,
                requested,
                maximum,
            } => write!(
                f,
                "refusing to allocate {requested} bytes for tensor {name:?}; caller limit is {maximum} bytes"
            ),
        }
    }
}

impl std::error::Error for SafetensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct TensorIndex {
    model_dir: PathBuf,
    shards: Vec<ShardInfo>,
    files: HashMap<PathBuf, File>,
    #[cfg(target_os = "linux")]
    direct_files: HashMap<PathBuf, File>,
    tensors: HashMap<String, TensorInfo>,
    names: Vec<String>,
}

impl TensorIndex {
    pub fn open(model_dir: &Path) -> Result<Self, SafetensorError> {
        let entries = fs::read_dir(model_dir).map_err(|source| SafetensorError::Io {
            path: model_dir.to_path_buf(),
            source,
        })?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| SafetensorError::Io {
                path: model_dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
                paths.push(path);
            }
        }
        paths.sort();
        if paths.is_empty() {
            return Err(SafetensorError::NoShards(model_dir.to_path_buf()));
        }

        let mut shards = Vec::with_capacity(paths.len());
        let mut files = HashMap::with_capacity(paths.len());
        #[cfg(target_os = "linux")]
        let mut direct_files = HashMap::with_capacity(paths.len());
        let mut tensors: HashMap<String, TensorInfo> = HashMap::new();
        for path in paths {
            let (shard, entries, file) = read_shard_header(&path)?;
            for tensor in entries {
                if let Some(first) = tensors.get(&tensor.name) {
                    return Err(SafetensorError::DuplicateTensor {
                        name: tensor.name,
                        first: first.shard.clone(),
                        second: tensor.shard,
                    });
                }
                tensors.insert(tensor.name.clone(), tensor);
            }
            #[cfg(target_os = "linux")]
            if let Ok(direct) = OpenOptions::new()
                .read(true)
                .custom_flags(O_DIRECT)
                .open(&path)
            {
                direct_files.insert(path.clone(), direct);
            }
            files.insert(path, file);
            shards.push(shard);
        }
        let mut names: Vec<String> = tensors.keys().cloned().collect();
        names.sort();
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            shards,
            files,
            #[cfg(target_os = "linux")]
            direct_files,
            tensors,
            names,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn shards(&self) -> &[ShardInfo] {
        &self.shards
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.names.iter().map(|name| &self.tensors[name])
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn require(&self, name: &str) -> Result<&TensorInfo, SafetensorError> {
        self.get(name)
            .ok_or_else(|| SafetensorError::MissingTensor(name.to_owned()))
    }

    pub fn total_file_bytes(&self) -> u64 {
        self.shards.iter().map(|shard| shard.file_len).sum()
    }

    pub fn total_payload_bytes(&self) -> u64 {
        self.shards.iter().map(|shard| shard.payload_bytes).sum()
    }

    /// Reads a bounded byte range within a tensor. It never maps or reads the rest of the shard.
    pub fn read_range(
        &self,
        name: &str,
        relative_offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, SafetensorError> {
        if len > MAX_SINGLE_READ_BYTES {
            return Err(SafetensorError::ReadTooLarge {
                requested: len,
                maximum: MAX_SINGLE_READ_BYTES,
            });
        }
        let tensor = self.require(name)?;
        let len_u64 = u64::try_from(len).map_err(|_| SafetensorError::ReadTooLarge {
            requested: len,
            maximum: MAX_SINGLE_READ_BYTES,
        })?;
        let end = relative_offset
            .checked_add(len_u64)
            .ok_or_else(|| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!("range overflow while reading tensor {name:?}"),
            })?;
        if end > tensor.data_len {
            return Err(SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!(
                    "range [{relative_offset}, {end}) exceeds tensor {name:?} length {}",
                    tensor.data_len
                ),
            });
        }

        let absolute = tensor
            .data_offset
            .checked_add(relative_offset)
            .ok_or_else(|| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!("absolute offset overflow while reading tensor {name:?}"),
            })?;
        self.read_owned_exact_at(tensor, absolute, len)
    }

    /// Reads a bounded range into an equally sized owned buffer when one is available.
    ///
    /// Routed-expert caches repeatedly replace same-shaped tensors. Reusing an evicted tensor's
    /// initialized allocation avoids allocator churn and first-touch page faults; a mismatched
    /// buffer falls back to the ordinary uninitialized positional-read allocation.
    pub(crate) fn read_range_reusing(
        &self,
        name: &str,
        relative_offset: u64,
        len: usize,
        mut output: Vec<u8>,
    ) -> Result<Vec<u8>, SafetensorError> {
        if output.len() != len {
            return self.read_range(name, relative_offset, len);
        }
        if len > MAX_SINGLE_READ_BYTES {
            return Err(SafetensorError::ReadTooLarge {
                requested: len,
                maximum: MAX_SINGLE_READ_BYTES,
            });
        }
        let tensor = self.require(name)?;
        let len_u64 = u64::try_from(len).map_err(|_| SafetensorError::ReadTooLarge {
            requested: len,
            maximum: MAX_SINGLE_READ_BYTES,
        })?;
        let end = relative_offset
            .checked_add(len_u64)
            .ok_or_else(|| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!("range overflow while reading tensor {name:?}"),
            })?;
        if end > tensor.data_len {
            return Err(SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!(
                    "range [{relative_offset}, {end}) exceeds tensor {name:?} length {}",
                    tensor.data_len
                ),
            });
        }
        let absolute = tensor
            .data_offset
            .checked_add(relative_offset)
            .ok_or_else(|| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!("absolute offset overflow while reading tensor {name:?}"),
            })?;
        self.read_exact_at(tensor, absolute, &mut output)?;
        Ok(output)
    }

    /// Reads through the ordinary buffered descriptor while retaining a direct-I/O-compatible
    /// allocation. Hy4 prefill uses this path so switching to direct reads for decode does not
    /// require reallocating every cached and recycled expert tensor.
    pub(crate) fn read_range_aligned_reusing(
        &self,
        name: &str,
        relative_offset: u64,
        len: usize,
        reuse: Option<ReadBuffer>,
    ) -> Result<ReadBuffer, SafetensorError> {
        #[cfg(target_os = "linux")]
        let mut reuse = reuse;
        #[cfg(target_os = "linux")]
        if direct_expert_io_enabled() && len != 0 {
            if len > MAX_SINGLE_READ_BYTES {
                return Err(SafetensorError::ReadTooLarge {
                    requested: len,
                    maximum: MAX_SINGLE_READ_BYTES,
                });
            }
            let tensor = self.require(name)?;
            let len_u64 = u64::try_from(len).map_err(|_| SafetensorError::ReadTooLarge {
                requested: len,
                maximum: MAX_SINGLE_READ_BYTES,
            })?;
            let end =
                relative_offset
                    .checked_add(len_u64)
                    .ok_or_else(|| SafetensorError::Invalid {
                        path: tensor.shard.clone(),
                        reason: format!("range overflow while reading tensor {name:?}"),
                    })?;
            if end > tensor.data_len {
                return Err(SafetensorError::Invalid {
                    path: tensor.shard.clone(),
                    reason: format!(
                        "range [{relative_offset}, {end}) exceeds tensor {name:?} length {}",
                        tensor.data_len
                    ),
                });
            }
            let absolute = tensor
                .data_offset
                .checked_add(relative_offset)
                .ok_or_else(|| SafetensorError::Invalid {
                    path: tensor.shard.clone(),
                    reason: format!("absolute offset overflow while reading tensor {name:?}"),
                })?;
            if self.direct_files.contains_key(&tensor.shard) {
                let prefix = usize::try_from(absolute & (DIRECT_ALIGNMENT as u64 - 1))
                    .expect("a page-alignment prefix fits usize");
                let required = prefix
                    .checked_add(len)
                    .ok_or(SafetensorError::ReadTooLarge {
                        requested: len,
                        maximum: MAX_SINGLE_READ_BYTES,
                    })?;
                let direct_reuse = match reuse.take() {
                    Some(ReadBuffer::Direct(buffer)) => Some(buffer),
                    Some(buffer) => {
                        reuse = Some(buffer);
                        None
                    }
                    None => None,
                };
                let buffer = direct_reuse.or_else(|| DirectReadBuffer::with_capacity(required));
                if let Some(mut buffer) = buffer {
                    if buffer.ensure_capacity(required) {
                        let file = self.files.get(&tensor.shard).ok_or_else(|| {
                            SafetensorError::Invalid {
                                path: tensor.shard.clone(),
                                reason: "indexed shard has no retained file handle".to_owned(),
                            }
                        })?;
                        let base_offset =
                            i64::try_from(absolute).map_err(|_| SafetensorError::Invalid {
                                path: tensor.shard.clone(),
                                reason: "positional read offset exceeds the platform limit"
                                    .to_owned(),
                            })?;
                        let mut filled = 0usize;
                        while filled < len {
                            let offset =
                                base_offset.checked_add(filled as i64).ok_or_else(|| {
                                    SafetensorError::Invalid {
                                        path: tensor.shard.clone(),
                                        reason: "positional read offset overflows i64".to_owned(),
                                    }
                                })?;
                            unsafe extern "C" {
                                fn pread(
                                    file_descriptor: i32,
                                    buffer: *mut core::ffi::c_void,
                                    count: usize,
                                    offset: i64,
                                ) -> isize;
                            }
                            // SAFETY: `prefix + len <= capacity`; bytes before the logical range
                            // need not be initialized for a buffered read and are never exposed.
                            let result = unsafe {
                                pread(
                                    file.as_raw_fd(),
                                    buffer
                                        .allocation_ptr()
                                        .add(prefix + filled)
                                        .cast::<core::ffi::c_void>(),
                                    len - filled,
                                    offset,
                                )
                            };
                            if result < 0 {
                                let source = io::Error::last_os_error();
                                if source.kind() == io::ErrorKind::Interrupted {
                                    continue;
                                }
                                return Err(SafetensorError::Io {
                                    path: tensor.shard.clone(),
                                    source,
                                });
                            }
                            if result == 0 {
                                return Err(SafetensorError::Io {
                                    path: tensor.shard.clone(),
                                    source: io::Error::new(
                                        io::ErrorKind::UnexpectedEof,
                                        "positional tensor read reached EOF",
                                    ),
                                });
                            }
                            let initialized =
                                usize::try_from(result).map_err(|_| SafetensorError::Invalid {
                                    path: tensor.shard.clone(),
                                    reason: "positional read returned a negative byte count"
                                        .to_owned(),
                                })?;
                            if initialized > len - filled {
                                return Err(SafetensorError::Invalid {
                                    path: tensor.shard.clone(),
                                    reason: "positional read returned more bytes than requested"
                                        .to_owned(),
                                });
                            }
                            filled += initialized;
                        }
                        buffer.set_range(prefix, len);
                        return Ok(ReadBuffer::Direct(buffer));
                    }
                }
            }
        }
        let output = match reuse {
            Some(output) => {
                self.read_range_reusing(name, relative_offset, len, output.into_vec())?
            }
            None => self.read_range(name, relative_offset, len)?,
        };
        Ok(ReadBuffer::Heap(output))
    }

    /// Reads a large immutable range without filling the OS page cache when Linux and the backing
    /// filesystem support direct I/O. A per-thread aligned bounce allocation is reused, then the
    /// bytes are copied into the caller's ordinary owned buffer. Unsupported or short direct reads
    /// transparently fall back to the existing buffered positional path.
    pub(crate) fn read_range_direct_reusing(
        &self,
        name: &str,
        relative_offset: u64,
        len: usize,
        reuse: Option<ReadBuffer>,
    ) -> Result<ReadBuffer, SafetensorError> {
        #[cfg(target_os = "linux")]
        let mut reuse = reuse;
        #[cfg(target_os = "linux")]
        if direct_expert_io_enabled() {
            let tensor = self.require(name)?;
            let len_u64 = u64::try_from(len).map_err(|_| SafetensorError::ReadTooLarge {
                requested: len,
                maximum: MAX_SINGLE_READ_BYTES,
            })?;
            let end =
                relative_offset
                    .checked_add(len_u64)
                    .ok_or_else(|| SafetensorError::Invalid {
                        path: tensor.shard.clone(),
                        reason: format!("range overflow while reading tensor {name:?}"),
                    })?;
            if len <= MAX_SINGLE_READ_BYTES && end <= tensor.data_len && len != 0 {
                let absolute =
                    tensor
                        .data_offset
                        .checked_add(relative_offset)
                        .ok_or_else(|| SafetensorError::Invalid {
                            path: tensor.shard.clone(),
                            reason: format!(
                                "absolute offset overflow while reading tensor {name:?}"
                            ),
                        })?;
                if let Some(file) = self.direct_files.get(&tensor.shard) {
                    let aligned = absolute & !(DIRECT_ALIGNMENT as u64 - 1);
                    let prefix = usize::try_from(absolute - aligned)
                        .expect("a page-alignment prefix fits usize");
                    let required =
                        prefix
                            .checked_add(len)
                            .ok_or(SafetensorError::ReadTooLarge {
                                requested: len,
                                maximum: MAX_SINGLE_READ_BYTES,
                            })?;
                    let read_len = required
                        .checked_add(DIRECT_ALIGNMENT - 1)
                        .map(|value| value / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT)
                        .ok_or(SafetensorError::ReadTooLarge {
                            requested: len,
                            maximum: MAX_SINGLE_READ_BYTES,
                        })?;
                    let direct_reuse = match reuse.take() {
                        Some(ReadBuffer::Direct(buffer)) => Some(buffer),
                        Some(buffer) => {
                            reuse = Some(buffer);
                            None
                        }
                        None => None,
                    };
                    let buffer = direct_reuse.or_else(|| DirectReadBuffer::with_capacity(read_len));
                    if let Some(mut buffer) = buffer {
                        if buffer.ensure_capacity(read_len) {
                            if let Ok(offset) = i64::try_from(aligned) {
                                unsafe extern "C" {
                                    fn pread(
                                        file_descriptor: i32,
                                        buffer: *mut core::ffi::c_void,
                                        count: usize,
                                        offset: i64,
                                    ) -> isize;
                                }
                                let result = loop {
                                    // SAFETY: the owned allocation is `read_len` bytes long and
                                    // page-aligned; the descriptor and aligned offset remain valid.
                                    let result = unsafe {
                                        pread(
                                            file.as_raw_fd(),
                                            buffer.allocation_ptr().cast::<core::ffi::c_void>(),
                                            read_len,
                                            offset,
                                        )
                                    };
                                    if result < 0
                                        && io::Error::last_os_error().kind()
                                            == io::ErrorKind::Interrupted
                                    {
                                        continue;
                                    }
                                    break result;
                                };
                                if result >= 0 && (result as usize) >= required {
                                    buffer.set_range(prefix, len);
                                    return Ok(ReadBuffer::Direct(buffer));
                                }
                            }
                        }
                    }
                }
            }
        }
        let output = match reuse {
            Some(output) => {
                self.read_range_reusing(name, relative_offset, len, output.into_vec())?
            }
            None => self.read_range(name, relative_offset, len)?,
        };
        Ok(ReadBuffer::Heap(output))
    }

    /// Reads one complete tensor through bounded chunks, subject to an explicit caller budget.
    ///
    /// This is used by the resident-core and expert loaders. The 64 MiB I/O bound remains in
    /// force for every individual read; `maximum_bytes` controls the final owned allocation.
    pub fn read_tensor_bounded(
        &self,
        name: &str,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, SafetensorError> {
        let tensor = self.require(name)?;
        if tensor.data_len > maximum_bytes {
            return Err(SafetensorError::AllocationTooLarge {
                name: name.to_owned(),
                requested: tensor.data_len,
                maximum: maximum_bytes,
            });
        }
        let length =
            usize::try_from(tensor.data_len).map_err(|_| SafetensorError::AllocationTooLarge {
                name: name.to_owned(),
                requested: tensor.data_len,
                maximum: maximum_bytes,
            })?;
        self.read_owned_exact_at(tensor, tensor.data_offset, length)
    }

    /// Reads complete tensors in physical shard order and returns them in caller order.
    ///
    /// This is intended for related weights such as the three projections of one expert. It
    /// keeps their independent owned buffers, but avoids alternating between distant payload and
    /// scale regions in a shard.
    pub fn read_tensors_bounded(
        &self,
        names: &[&str],
        maximum_bytes: u64,
    ) -> Result<Vec<Vec<u8>>, SafetensorError> {
        let mut total = 0u64;
        let mut tensors = Vec::with_capacity(names.len());
        for &name in names {
            let tensor = self.require(name)?;
            total = total.checked_add(tensor.data_len).ok_or_else(|| {
                SafetensorError::AllocationTooLarge {
                    name: "tensor batch".to_owned(),
                    requested: u64::MAX,
                    maximum: maximum_bytes,
                }
            })?;
            tensors.push(tensor);
        }
        if total > maximum_bytes {
            return Err(SafetensorError::AllocationTooLarge {
                name: "tensor batch".to_owned(),
                requested: total,
                maximum: maximum_bytes,
            });
        }

        // Validate every allocation length before issuing any payload I/O. Allocate each buffer
        // only when its physical-order read starts, and let the owned reader initialize it once.
        let mut lengths = Vec::with_capacity(tensors.len());
        for (&name, tensor) in names.iter().zip(&tensors) {
            let length = usize::try_from(tensor.data_len).map_err(|_| {
                SafetensorError::AllocationTooLarge {
                    name: name.to_owned(),
                    requested: tensor.data_len,
                    maximum: maximum_bytes,
                }
            })?;
            lengths.push(length);
        }

        let mut physical_order = (0..tensors.len()).collect::<Vec<_>>();
        physical_order.sort_by(|&left, &right| {
            tensors[left]
                .shard
                .cmp(&tensors[right].shard)
                .then_with(|| tensors[left].data_offset.cmp(&tensors[right].data_offset))
        });
        let mut output = vec![Vec::new(); tensors.len()];
        for index in physical_order {
            let tensor = tensors[index];
            output[index] = self.read_owned_exact_at(tensor, tensor.data_offset, lengths[index])?;
        }
        Ok(output)
    }

    /// Overwrites caller-owned tensor buffers in physical shard order and returns them in caller
    /// order. This combines the seek-friendly ordering of [`Self::read_tensors_bounded`] with the
    /// allocation and page reuse needed by repeatedly evicted expert caches.
    pub(crate) fn read_tensors_bounded_reusing(
        &self,
        names: &[&str],
        mut reuse: Vec<Vec<u8>>,
        maximum_bytes: u64,
    ) -> Result<Vec<Vec<u8>>, SafetensorError> {
        if names.len() != reuse.len() {
            return Err(SafetensorError::Invalid {
                path: self.model_dir.clone(),
                reason: format!(
                    "received {} reusable buffers for {} tensors",
                    reuse.len(),
                    names.len()
                ),
            });
        }

        let mut total = 0u64;
        let mut tensors = Vec::with_capacity(names.len());
        let mut lengths = Vec::with_capacity(names.len());
        for &name in names {
            let tensor = self.require(name)?;
            total = total.checked_add(tensor.data_len).ok_or_else(|| {
                SafetensorError::AllocationTooLarge {
                    name: "reusable tensor batch".to_owned(),
                    requested: u64::MAX,
                    maximum: maximum_bytes,
                }
            })?;
            let length = usize::try_from(tensor.data_len).map_err(|_| {
                SafetensorError::AllocationTooLarge {
                    name: name.to_owned(),
                    requested: tensor.data_len,
                    maximum: maximum_bytes,
                }
            })?;
            tensors.push(tensor);
            lengths.push(length);
        }
        if total > maximum_bytes {
            return Err(SafetensorError::AllocationTooLarge {
                name: "reusable tensor batch".to_owned(),
                requested: total,
                maximum: maximum_bytes,
            });
        }

        let mut physical_order = (0..tensors.len()).collect::<Vec<_>>();
        physical_order.sort_by(|&left, &right| {
            tensors[left]
                .shard
                .cmp(&tensors[right].shard)
                .then_with(|| tensors[left].data_offset.cmp(&tensors[right].data_offset))
        });
        let mut output = (0..tensors.len()).map(|_| None).collect::<Vec<_>>();
        for index in physical_order {
            let tensor = tensors[index];
            let length = lengths[index];
            let mut buffer = std::mem::take(&mut reuse[index]);
            if buffer.len() == length {
                self.read_exact_at(tensor, tensor.data_offset, &mut buffer)?;
                output[index] = Some(buffer);
            } else {
                output[index] =
                    Some(self.read_owned_exact_at(tensor, tensor.data_offset, length)?);
            }
        }
        Ok(output
            .into_iter()
            .map(|buffer| buffer.expect("every reusable batch entry was read"))
            .collect())
    }

    fn read_exact_at(
        &self,
        tensor: &TensorInfo,
        absolute: u64,
        output: &mut [u8],
    ) -> Result<(), SafetensorError> {
        #[cfg(unix)]
        {
            let file = self
                .files
                .get(&tensor.shard)
                .ok_or_else(|| SafetensorError::Invalid {
                    path: tensor.shard.clone(),
                    reason: "indexed shard has no retained file handle".to_owned(),
                })?;
            file.read_exact_at(output, absolute)
                .map_err(|source| SafetensorError::Io {
                    path: tensor.shard.clone(),
                    source,
                })
        }
        #[cfg(not(unix))]
        {
            let mut file = File::open(&tensor.shard).map_err(|source| SafetensorError::Io {
                path: tensor.shard.clone(),
                source,
            })?;
            file.seek(SeekFrom::Start(absolute))
                .map_err(|source| SafetensorError::Io {
                    path: tensor.shard.clone(),
                    source,
                })?;
            file.read_exact(output)
                .map_err(|source| SafetensorError::Io {
                    path: tensor.shard.clone(),
                    source,
                })
        }
    }

    /// Allocates a positional-read destination without first zero-filling pages that `pread` will
    /// immediately overwrite. Conversion to initialized bytes happens only after the exact read
    /// succeeds; `MaybeUninit<u8>` makes every early-error path safe to drop. Full tensors may be
    /// larger than the per-read bound, but each syscall reads at most 64 MiB into this allocation.
    #[cfg(unix)]
    fn read_owned_exact_at(
        &self,
        tensor: &TensorInfo,
        absolute: u64,
        length: usize,
    ) -> Result<Vec<u8>, SafetensorError> {
        unsafe extern "C" {
            fn pread(
                file_descriptor: i32,
                buffer: *mut core::ffi::c_void,
                count: usize,
                offset: i64,
            ) -> isize;
        }
        let file = self
            .files
            .get(&tensor.shard)
            .ok_or_else(|| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: "indexed shard has no retained file handle".to_owned(),
            })?;
        let base_offset = i64::try_from(absolute).map_err(|_| SafetensorError::Invalid {
            path: tensor.shard.clone(),
            reason: "positional read offset exceeds the platform limit".to_owned(),
        })?;
        let mut output = Vec::<MaybeUninit<u8>>::new();
        output
            .try_reserve_exact(length)
            .map_err(|error| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!("cannot reserve {length} positional-read bytes: {error}"),
            })?;
        // SAFETY: every element type is `MaybeUninit<u8>`, for which an uninitialized value is
        // valid. No `u8` view is created until the loop has initialized the entire range.
        unsafe {
            output.set_len(length);
        }
        let mut filled = 0usize;
        while filled < length {
            let filled_offset = i64::try_from(filled).map_err(|_| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: "positional read offset exceeds i64".to_owned(),
            })?;
            let offset =
                base_offset
                    .checked_add(filled_offset)
                    .ok_or_else(|| SafetensorError::Invalid {
                        path: tensor.shard.clone(),
                        reason: "positional read offset overflows i64".to_owned(),
                    })?;
            let chunk = (length - filled).min(MAX_SINGLE_READ_BYTES);
            // SAFETY: `filled < length`; the destination spans at most the remaining allocation,
            // and `pread` neither retains the pointer nor changes the file cursor.
            let result = unsafe {
                pread(
                    file.as_raw_fd(),
                    output.as_mut_ptr().add(filled).cast::<core::ffi::c_void>(),
                    chunk,
                    offset,
                )
            };
            if result < 0 {
                let source = io::Error::last_os_error();
                if source.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(SafetensorError::Io {
                    path: tensor.shard.clone(),
                    source,
                });
            }
            if result == 0 {
                return Err(SafetensorError::Io {
                    path: tensor.shard.clone(),
                    source: io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "positional tensor read reached EOF",
                    ),
                });
            }
            let initialized = usize::try_from(result).map_err(|_| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: "positional read returned a negative byte count".to_owned(),
            })?;
            if initialized > chunk {
                return Err(SafetensorError::Invalid {
                    path: tensor.shard.clone(),
                    reason: "positional read returned more bytes than requested".to_owned(),
                });
            }
            filled += initialized;
        }
        let mut output = ManuallyDrop::new(output);
        // SAFETY: the exact-read loop initialized all `length` bytes. `MaybeUninit<u8>` and `u8`
        // have identical layout, and ownership of the original allocation transfers to the new
        // vector without changing its pointer, length, or capacity.
        Ok(unsafe {
            Vec::from_raw_parts(
                output.as_mut_ptr().cast::<u8>(),
                output.len(),
                output.capacity(),
            )
        })
    }

    #[cfg(not(unix))]
    fn read_owned_exact_at(
        &self,
        tensor: &TensorInfo,
        absolute: u64,
        length: usize,
    ) -> Result<Vec<u8>, SafetensorError> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|error| SafetensorError::Invalid {
                path: tensor.shard.clone(),
                reason: format!("cannot reserve {length} positional-read bytes: {error}"),
            })?;
        output.resize(length, 0);
        for (chunk_index, chunk) in output.chunks_mut(MAX_SINGLE_READ_BYTES).enumerate() {
            let offset = absolute
                .checked_add((chunk_index * MAX_SINGLE_READ_BYTES) as u64)
                .ok_or_else(|| SafetensorError::Invalid {
                    path: tensor.shard.clone(),
                    reason: "positional read offset overflows u64".to_owned(),
                })?;
            self.read_exact_at(tensor, offset, chunk)?;
        }
        Ok(output)
    }
}

fn read_shard_header(path: &Path) -> Result<(ShardInfo, Vec<TensorInfo>, File), SafetensorError> {
    let mut file = File::open(path).map_err(|source| SafetensorError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if file_len < 8 {
        return invalid(path, "file is shorter than the 8-byte header length");
    }

    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return invalid(
            path,
            format!("header length {header_len} is outside [1, {MAX_HEADER_BYTES}]"),
        );
    }
    let data_base = 8u64
        .checked_add(header_len)
        .ok_or_else(|| SafetensorError::Invalid {
            path: path.to_path_buf(),
            reason: "header offset overflow".to_owned(),
        })?;
    if data_base > file_len {
        return invalid(
            path,
            format!("header ends at byte {data_base}, beyond file length {file_len}"),
        );
    }

    let header_size = usize::try_from(header_len).map_err(|_| SafetensorError::Invalid {
        path: path.to_path_buf(),
        reason: "header length cannot fit in memory on this platform".to_owned(),
    })?;
    let mut header = vec![0u8; header_size];
    file.read_exact(&mut header)
        .map_err(|source| SafetensorError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let root: RawHeader =
        serde_json::from_slice(&header).map_err(|source| SafetensorError::Invalid {
            path: path.to_path_buf(),
            reason: format!("header is not valid JSON: {source}"),
        })?;

    let payload_len = file_len - data_base;
    let metadata = root.metadata;
    let mut tensors = Vec::with_capacity(root.tensors.len());
    for (name, descriptor) in root.tensors {
        let dtype = DType::parse(&descriptor.dtype);
        if matches!(dtype, DType::Unknown(_)) {
            return invalid(
                path,
                format!(
                    "tensor {name:?} uses unsupported dtype {:?}",
                    descriptor.dtype
                ),
            );
        }
        let shape = descriptor.shape;
        let declared_elements = shape
            .iter()
            .try_fold(1u64, |acc, &dimension| acc.checked_mul(dimension))
            .ok_or_else(|| SafetensorError::Invalid {
                path: path.to_path_buf(),
                reason: format!("tensor {name:?} shape product overflows u64"),
            })?;

        let [start, end] = descriptor.data_offsets;
        if start > end || end > payload_len {
            return invalid(
                path,
                format!(
                    "tensor {name:?} range [{start}, {end}) is outside payload length {payload_len}"
                ),
            );
        }
        let data_len = end - start;
        if let Some(element_bytes) = dtype.element_bytes() {
            let expected = declared_elements
                .checked_mul(element_bytes)
                .ok_or_else(|| SafetensorError::Invalid {
                    path: path.to_path_buf(),
                    reason: format!("tensor {name:?} byte length overflows u64"),
                })?;
            if expected != data_len {
                return invalid(
                    path,
                    format!(
                        "tensor {name:?} dtype/shape imply {expected} bytes, range contains {data_len}"
                    ),
                );
            }
        }
        tensors.push(TensorInfo {
            name,
            dtype,
            shape,
            shard: path.to_path_buf(),
            data_offset: data_base + start,
            data_len,
            declared_elements,
        });
    }

    let mut ranges: Vec<(u64, u64, &str)> = tensors
        .iter()
        .map(|tensor| {
            let relative = tensor.data_offset - data_base;
            (relative, relative + tensor.data_len, tensor.name.as_str())
        })
        .collect();
    ranges.sort_by_key(|range| range.0);
    let mut covered_until = 0u64;
    for &(start, end, name) in &ranges {
        if start != covered_until {
            return invalid(
                path,
                format!(
                    "tensor {name:?} starts at {start}, but strict contiguous payload coverage expected {covered_until}"
                ),
            );
        }
        covered_until = end;
    }
    if covered_until != payload_len {
        return invalid(
            path,
            format!(
                "indexed tensor payload ends at {covered_until}, but shard payload has {payload_len} bytes"
            ),
        );
    }

    let payload_bytes = tensors.iter().map(|tensor| tensor.data_len).sum();
    let info = ShardInfo {
        path: path.to_path_buf(),
        file_len,
        header_len,
        tensor_count: tensors.len(),
        payload_bytes,
        metadata,
    };
    Ok((info, tensors, file))
}

fn invalid<T>(path: &Path, reason: impl Into<String>) -> Result<T, SafetensorError> {
    Err(SafetensorError::Invalid {
        path: path.to_path_buf(),
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir(name: &str) -> PathBuf {
        crate::test_support::temp_dir(&format!("urbilateria_{name}"))
    }

    fn write_fixture(path: &Path) {
        let f32_data: Vec<u8> = [1.0f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let u8_data = vec![7u8, 8, 9];
        let mut header = serde_json::json!({
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "U8", "shape": [3], "data_offsets": [8, 11]},
            "__metadata__": {"creator": "test"}
        })
        .to_string()
        .into_bytes();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(f32_data);
        bytes.extend(u8_data);
        fs::write(path, bytes).unwrap();
    }

    fn write_raw_fixture(path: &Path, raw_header: &str, payload: &[u8]) {
        let mut header = raw_header.as_bytes().to_vec();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn indexes_headers_without_reading_whole_payload() {
        let dir = fixture_dir("index");
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("model.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        assert_eq!(index.shards().len(), 1);
        assert_eq!(index.names().collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(index.require("a").unwrap().dtype, DType::F32);
        assert_eq!(index.read_range("b", 1, 2).unwrap(), vec![8, 9]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn same_sized_range_read_reuses_the_owned_allocation() {
        let dir = fixture_dir("reuse");
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("model.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let buffer = vec![0xaau8; 2];
        let pointer = buffer.as_ptr();
        let output = index.read_range_reusing("b", 1, 2, buffer).unwrap();
        assert_eq!(output, [8, 9]);
        assert_eq!(output.as_ptr(), pointer);
        let direct_pointer = output.as_ptr();
        let direct = index
            .read_range_direct_reusing("b", 1, 2, Some(output.into()))
            .unwrap();
        assert_eq!(direct.as_ref(), [8, 9]);
        match &direct {
            ReadBuffer::Heap(_) => assert_eq!(direct.as_ptr(), direct_pointer),
            #[cfg(target_os = "linux")]
            ReadBuffer::Direct(_) => {
                let expected = (index.require("b").unwrap().data_offset + 1)
                    % u64::try_from(DIRECT_ALIGNMENT).unwrap();
                assert_eq!(
                    direct.as_ptr() as usize % DIRECT_ALIGNMENT,
                    expected as usize
                );
            }
        }
        let direct_pointer = direct.as_ptr();
        let direct = index
            .read_range_direct_reusing("b", 1, 2, Some(direct))
            .unwrap();
        assert_eq!(direct.as_ref(), [8, 9]);
        assert_eq!(direct.as_ptr(), direct_pointer);
        let was_direct = !matches!(direct, ReadBuffer::Heap(_));
        let aligned_pointer = direct.as_ptr();
        let aligned = index
            .read_range_aligned_reusing("b", 1, 2, Some(direct))
            .unwrap();
        assert_eq!(aligned.as_ref(), [8, 9]);
        if was_direct || matches!(aligned, ReadBuffer::Heap(_)) {
            assert_eq!(aligned.as_ptr(), aligned_pointer);
        }
        assert_eq!(
            index
                .read_range_direct_reusing("b", 0, 3, None)
                .unwrap()
                .as_ref(),
            [7, 8, 9]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn batch_reads_in_physical_order_but_returns_caller_order() {
        let dir = fixture_dir("batch_order");
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("model.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let tensors = index.read_tensors_bounded(&["b", "a"], 11).unwrap();
        assert_eq!(tensors[0], vec![7, 8, 9]);
        assert_eq!(
            tensors[1],
            [1.0f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            index.read_tensors_bounded(&["a", "b"], 10),
            Err(SafetensorError::AllocationTooLarge {
                requested: 11,
                maximum: 10,
                ..
            })
        ));

        let b_buffer = vec![0xaau8; 3];
        let a_buffer = vec![0xbbu8; 8];
        let pointers = [b_buffer.as_ptr(), a_buffer.as_ptr()];
        let tensors = index
            .read_tensors_bounded_reusing(&["b", "a"], vec![b_buffer, a_buffer], 11)
            .unwrap();
        assert_eq!(tensors[0], vec![7, 8, 9]);
        assert_eq!(
            tensors[1],
            [1.0f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(tensors[0].as_ptr(), pointers[0]);
        assert_eq!(tensors[1].as_ptr(), pointers[1]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bounded_tensor_reads_cover_multiple_io_chunks_without_gaps() {
        use std::io::{Seek, SeekFrom, Write};

        let dir = fixture_dir("multi_chunk");
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("model.safetensors");
        let length = MAX_SINGLE_READ_BYTES + 19;
        let header = serde_json::json!({
            "large": {"dtype": "U8", "shape": [length], "data_offsets": [0, length]}
        })
        .to_string();
        write_raw_fixture(&shard, &header, &[]);
        let mut file = File::options().write(true).open(&shard).unwrap();
        let data_offset = file.metadata().unwrap().len();
        // A sparse fixture exercises the real 64 MiB boundary without allocating its payload
        // during construction. Place nonzero markers on both sides to expose gaps or overlap.
        file.set_len(data_offset + length as u64).unwrap();
        file.seek(SeekFrom::Start(data_offset)).unwrap();
        file.write_all(&[1, 2, 3]).unwrap();
        file.seek(SeekFrom::Start(
            data_offset + MAX_SINGLE_READ_BYTES as u64 - 3,
        ))
        .unwrap();
        file.write_all(&[11, 22, 33, 44, 55, 66]).unwrap();
        file.seek(SeekFrom::Start(data_offset + length as u64 - 2))
            .unwrap();
        file.write_all(&[254, 255]).unwrap();
        drop(file);
        let index = TensorIndex::open(&dir).unwrap();
        let check = |bytes: &[u8]| {
            assert_eq!(bytes.len(), length);
            assert_eq!(&bytes[..3], [1, 2, 3]);
            assert!(bytes[3..MAX_SINGLE_READ_BYTES - 3]
                .iter()
                .all(|&byte| byte == 0));
            assert_eq!(
                &bytes[MAX_SINGLE_READ_BYTES - 3..MAX_SINGLE_READ_BYTES + 3],
                [11, 22, 33, 44, 55, 66]
            );
            assert!(bytes[MAX_SINGLE_READ_BYTES + 3..length - 2]
                .iter()
                .all(|&byte| byte == 0));
            assert_eq!(&bytes[length - 2..], [254, 255]);
        };
        check(&index.read_tensor_bounded("large", length as u64).unwrap());
        check(
            &index
                .read_tensors_bounded(&["large"], length as u64)
                .unwrap()[0],
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn full_tensor_reads_reject_budgets_before_io_and_discard_partial_eof() {
        let dir = fixture_dir("bounded_partial_eof");
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("model.safetensors");
        write_fixture(&shard);
        let index = TensorIndex::open(&dir).unwrap();
        // Keep all of a and the first byte of b: single reads partially initialize a buffer,
        // while batch reads finish a before failing partway through b.
        File::options()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(index.require("b").unwrap().data_offset + 1)
            .unwrap();
        assert!(matches!(
            index.read_tensor_bounded("b", 2),
            Err(SafetensorError::AllocationTooLarge {
                requested: 3,
                maximum: 2,
                ..
            })
        ));
        assert!(matches!(
            index.read_tensors_bounded(&["b", "a"], 10),
            Err(SafetensorError::AllocationTooLarge {
                requested: 11,
                maximum: 10,
                ..
            })
        ));
        assert!(matches!(
            index.read_tensor_bounded("b", 3),
            Err(SafetensorError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::UnexpectedEof
        ));
        assert!(matches!(
            index.read_tensors_bounded(&["b", "a"], 11),
            Err(SafetensorError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::UnexpectedEof
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn payload_reads_reuse_the_shard_handle_retained_by_the_index() {
        let dir = fixture_dir("retained_handle");
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("model.safetensors");
        let moved = dir.join("moved.safetensors");
        write_fixture(&shard);
        let index = TensorIndex::open(&dir).unwrap();
        fs::rename(&shard, &moved).unwrap();
        assert_eq!(index.read_range("b", 0, 3).unwrap(), vec![7, 8, 9]);
        drop(index);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn uninitialized_positional_buffer_never_escapes_an_early_eof() {
        let dir = fixture_dir("short_positional_read");
        fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("model.safetensors");
        write_fixture(&shard);
        let index = TensorIndex::open(&dir).unwrap();
        File::options()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(matches!(
            index.read_range("b", 0, 3),
            Err(SafetensorError::Io { source, .. })
                if source.kind() == io::ErrorKind::UnexpectedEof
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_out_of_bounds_ranges() {
        let dir = fixture_dir("bounds");
        fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("model.safetensors"));
        let index = TensorIndex::open(&dir).unwrap();
        let error = index.read_range("b", 2, 2).unwrap_err();
        assert!(error.to_string().contains("exceeds"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_duplicate_tensor_names_before_json_can_overwrite_them_strictly() {
        let dir = fixture_dir("duplicate_json_strict");
        fs::create_dir_all(&dir).unwrap();
        write_raw_fixture(
            &dir.join("model.safetensors"),
            "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",
            &[1, 2],
        );
        let error = TensorIndex::open(&dir).unwrap_err();
        assert!(error.to_string().contains("duplicate top-level key"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_unknown_dtype_and_unindexed_payload_gaps_strictly() {
        let unknown_dir = fixture_dir("unknown_dtype_strict");
        fs::create_dir_all(&unknown_dir).unwrap();
        write_raw_fixture(
            &unknown_dir.join("model.safetensors"),
            "{\"a\":{\"dtype\":\"F7\",\"shape\":[1],\"data_offsets\":[0,1]}}",
            &[0],
        );
        assert!(TensorIndex::open(&unknown_dir)
            .unwrap_err()
            .to_string()
            .contains("unsupported dtype"));
        fs::remove_dir_all(unknown_dir).unwrap();

        let gap_dir = fixture_dir("payload_gap_strict");
        fs::create_dir_all(&gap_dir).unwrap();
        write_raw_fixture(
            &gap_dir.join("model.safetensors"),
            "{\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",
            &[0, 1],
        );
        assert!(TensorIndex::open(&gap_dir)
            .unwrap_err()
            .to_string()
            .contains("contiguous payload"));
        fs::remove_dir_all(gap_dir).unwrap();
    }
}
