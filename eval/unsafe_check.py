# ur_check.py
#!/usr/bin/env python3
"""
Java→Rust Unsafe Check (UR / UR-LOC)

Preprocessing (ONLY):
- Strip Markdown code fences ```rust ... ``` (if present) and analyze the enclosed code.

Metrics (per-sample):
- UR     : 1.0 if any unsafe evidence exists, else 0.0
- UR-LOC : (# unsafe-hit lines) / (total lines)

Unsafe evidence = union of:
(A) Your Java→Rust hazard signals:
    - unsafe regions: unsafe {..}, unsafe fn/impl/trait {..}
    - extern "C" regions
    - raw pointers, pointer deref
    - unchecked ops, transmute, manual memory
    - libc:: usage
    - unsafe keyword (fallback)
(B) IRENE categories (added):
    - multi-line compiler directives: #![allow(...)] (spans)
    - unsafe C-stdio calls: printf/scanf (line-level)
"""

from __future__ import annotations
import re
from typing import Set, Dict, Any


# 0) Fence stripping
# ------------------

RE_CODE_FENCE_BLOCK = re.compile(
    r"```(?:\s*rust)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

def _strip_fences(text: str) -> str:
    """Remove fences and extract rust code (if fenced)."""
    m = RE_CODE_FENCE_BLOCK.search(text)
    return m.group(1) if m else text


# 1) Your Java→Rust unsafe / hazard signals (line-level)
# ------------------------------------------------------

RE_RAW_POINTER = re.compile(r"\*(const|mut)\b")  # *const T, *mut T
RE_POINTER_DEREF = re.compile(r"^\s*\*\s*[A-Za-z_][A-Za-z0-9_]*")  # *ptr (rough)

RE_UNCHECKED = re.compile(
    r"\b("
    r"unwrap_unchecked|get_unchecked|get_unchecked_mut|"
    r"assume_init|assume_init_mut|from_raw_parts|from_raw_parts_mut"
    r")\b"
)

RE_TRANSMUTE = re.compile(r"\b(?:core::mem::transmute|std::mem::transmute|mem::transmute)\b")
RE_MANUAL_MEMORY = re.compile(r"\b(alloc::alloc|alloc::dealloc|Box::from_raw|Vec::from_raw_parts)\b")

# Proxy: libc usage often implies FFI-ish patterns
RE_LIBC = re.compile(r"\blibc::[A-Za-z0-9_]+\b")

# Region starts
RE_EXTERN_C = re.compile(r'extern\s+"C"\b')

# Line-level unsafe keyword evidence (fallback)
RE_UNSAFE_KW = re.compile(r"\bunsafe\b")

# Specific unsafe-region starters for span extraction (minimal fix)
# Matches:
#   unsafe { ... }
#   unsafe fn ... { ... }
#   unsafe impl ... { ... }
#   unsafe trait ... { ... }
RE_UNSAFE_REGION_START = re.compile(r"(^|\s)\bunsafe\b\s*(\{|fn\b|impl\b|trait\b)")


# 2) IRENE categories (added)
# ---------------------------

# Multi-line compiler directives like #![allow(...)]
RE_ALLOW_DIRECTIVE_START = re.compile(r"^\s*#!\[\s*allow\b")

# Unsafe C stdio calls (IRENE used printf/scanf)
RE_UNSAFE_CSTDIO_CALLS = re.compile(r"\b(printf|scanf)\b")


# 3) Robust region extraction: supports `{` on later lines
# --------------------------------------------------------

def _extract_brace_regions(
    lines: list[str],
    start_pattern: re.Pattern,
    must_have_braces: bool = True,
) -> Set[int]:
    """
    Brace counting, robust to cases where '{' appears after the start line.

    - When start_pattern matches, we enter a "pending/inside" region.
    - We don't allow region to close until we've seen at least one '{'.
    - Then we close when brace_count returns to 0 or below.
    """
    region_lines: Set[int] = set()
    inside = False
    brace_count = 0
    seen_open = False

    for idx, line in enumerate(lines):
        if not inside and start_pattern.search(line):
            inside = True
            region_lines.add(idx)

            delta = line.count("{") - line.count("}")
            brace_count += delta
            if line.count("{") > 0:
                seen_open = True

            if seen_open and brace_count <= 0:
                inside = False
                brace_count = 0
                seen_open = False
            continue

        if inside:
            region_lines.add(idx)
            delta = line.count("{") - line.count("}")
            brace_count += delta
            if line.count("{") > 0:
                seen_open = True

            if (not must_have_braces) or seen_open:
                if brace_count <= 0:
                    inside = False
                    brace_count = 0
                    seen_open = False

    return region_lines


def _extract_allow_directive_lines(lines: list[str]) -> Set[int]:
    """
    Extract multi-line attribute directives that start with #![allow ...].
    Mimics IRENE's idea: include start line and any continuation lines until ')' appears.
    Handles common endings like ')]' as well.
    """
    directive_lines: Set[int] = set()
    inside = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not inside and RE_ALLOW_DIRECTIVE_START.search(stripped):
            inside = True
            directive_lines.add(i)
            # End if this line already closes the directive
            if ")" in stripped:
                inside = False
            continue

        if inside:
            directive_lines.add(i)
            if ")" in stripped:
                inside = False

    return directive_lines


# 4) Public API
# -------------

def analyze_unsafe_java_to_rust(code: str, return_details: bool = False):
    """
    Returns (default):
        (ur, urloc)
          ur    : 0.0 or 1.0
          urloc : unsafe_line_count / total_lines

    If return_details=True:
        (ur, urloc, details_dict)

    Notes:
    - unsafe lines are union of:
        * unsafe regions (unsafe fn/impl/trait/block bodies) via brace spans
        * extern "C" regions via brace spans
        * IRENE compiler directives: #![allow(...)] spans
        * dangerous API/pattern lines (raw pointers, unchecked, transmute, manual memory, libc)
        * unsafe C-stdio calls (printf/scanf)
        * 'unsafe' keyword lines as fallback evidence
    """
    code = _strip_fences(code)
    lines = code.strip().splitlines()
    total_lines = len(lines) if len(lines) > 0 else 1

    # Region spans
    unsafe_region_lines = _extract_brace_regions(lines, RE_UNSAFE_REGION_START, must_have_braces=True)
    extern_c_region_lines = _extract_brace_regions(lines, RE_EXTERN_C, must_have_braces=True)

    # IRENE directive spans
    allow_directive_lines = _extract_allow_directive_lines(lines)

    # Line-level signals
    raw_pointer_lines: Set[int] = set()
    pointer_deref_lines: Set[int] = set()
    unchecked_lines: Set[int] = set()
    transmute_lines: Set[int] = set()
    manual_memory_lines: Set[int] = set()
    libc_lines: Set[int] = set()
    unsafe_keyword_lines: Set[int] = set()
    unsafe_stdio_call_lines: Set[int] = set()

    for idx, line in enumerate(lines):
        if RE_RAW_POINTER.search(line):
            raw_pointer_lines.add(idx)
        if RE_POINTER_DEREF.search(line):
            pointer_deref_lines.add(idx)
        if RE_UNCHECKED.search(line):
            unchecked_lines.add(idx)
        if RE_TRANSMUTE.search(line):
            transmute_lines.add(idx)
        if RE_MANUAL_MEMORY.search(line):
            manual_memory_lines.add(idx)
        if RE_LIBC.search(line):
            libc_lines.add(idx)
        if RE_UNSAFE_CSTDIO_CALLS.search(line):
            unsafe_stdio_call_lines.add(idx)
        if RE_UNSAFE_KW.search(line):
            unsafe_keyword_lines.add(idx)

    unsafe_lines = set().union(
        # regions
        unsafe_region_lines,
        extern_c_region_lines,
        # IRENE compiler directives
        allow_directive_lines,
        # line-level signals
        raw_pointer_lines,
        pointer_deref_lines,
        unchecked_lines,
        transmute_lines,
        manual_memory_lines,
        libc_lines,
        unsafe_stdio_call_lines,
        # fallback
        unsafe_keyword_lines,
    )

    urloc = len(unsafe_lines) / total_lines
    ur = 1.0 if len(unsafe_lines) > 0 else 0.0

    if not return_details:
        return ur, urloc

    details: Dict[str, Any] = {
        "total_lines": total_lines,
        "unsafe_lines_count": len(unsafe_lines),
        "unsafe_line_indices": sorted(unsafe_lines),
        "regions": {
            "unsafe_region": sorted(unsafe_region_lines),
            "extern_c_region": sorted(extern_c_region_lines),
            "compiler_directives_allow": sorted(allow_directive_lines),
        },
        "signals": {
            "raw_pointer": sorted(raw_pointer_lines),
            "pointer_deref": sorted(pointer_deref_lines),
            "unchecked_ops": sorted(unchecked_lines),
            "transmute": sorted(transmute_lines),
            "manual_memory": sorted(manual_memory_lines),
            "libc_usage": sorted(libc_lines),
            "unsafe_cstdio_calls": sorted(unsafe_stdio_call_lines),
            "unsafe_keyword": sorted(unsafe_keyword_lines),
        },
    }
    return ur, urloc, details


# 5) CLI quick test
# -----------------

if __name__ == "__main__":
    rust_code = r"""
#![allow(dead_code,
         unused_imports)]

use std::alloc::{alloc, dealloc, Layout};
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::slice;

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

unsafe fn fill_unchecked(dst: *mut u8, n: usize, value: u8) {
    for i in 0..n {
        *dst.add(i) = value;
    }
}

fn main() {
    unsafe { printf(b"hi\0".as_ptr() as *const i8); }
}
"""
    ur, urloc, dbg = analyze_unsafe_java_to_rust(rust_code, return_details=True)
    print("UR:", ur)
    print("UR-LOC:", urloc)
    print("Unsafe lines:", dbg["unsafe_lines_count"], "/", dbg["total_lines"])
    print("Signals:", dbg["signals"])
