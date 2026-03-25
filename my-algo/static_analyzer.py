from tree_sitter import Language, Parser, Node
import json
import jsonlines
from typing import Dict, List, Callable, Any, Set, Tuple, Optional
from typing import Optional, Dict, List, Tuple

from pathlib import Path
import sys
import re


class JavaASTAnalyzer:
    """
    static analyzer for 5 parallel retrieval category:
      - io_parsing
      - arrays_indexing
      - numeric_mod_arith
      - datastructures
      - mut_ownership
    """

    CATS = ["io_parsing", "arrays_indexing", "numeric_mod_arith", "datastructures", "mut_ownership"]

    IO_METHODS = {"nextInt", "nextLong", "next", "readLine", "scanLong", "readInt", "nextDouble", "nextFloat"}
    IO_TYPES = {"Scanner", "BufferedReader", "InputStreamReader", "StringTokenizer", "PrintWriter",
                "FastScanner", "InputReader", "Scan", "Output", "Input"}

    DS_TYPES = {"ArrayList", "List", "HashMap", "Map", "HashSet", "Set", "PriorityQueue",
                "Queue", "Deque", "ArrayDeque", "LinkedHashMap", "TreeMap", "TreeSet"}

    # common ops for numeric category
    NUMERIC_OPS = {"%", "<<", ">>", "&", "|", "^"}
    MOD_HINTS = {"1000000007", "998244353", "1_000_000_007", "998_244_353", "1e9+7"}

    RUSTTYPE_PRIORITY = {
        "io_parsing": {
            "i64 then cast": 0,
            "Read+split_whitespace": 1,
            "Byte scanner": 2,
            "String out buffer": 3,
        },
        "arrays_indexing": {
            "usize index": 0,
            "n+1 sizing": 1,
            "bounds/offsets": 2,
            "2D mapping": 3,
        },
        "numeric_mod_arith": {
            "i128 mul": 0,
            "norm mod": 1,
            "mod_pow": 2,
            "bitmask typed": 3,
        },
        "datastructures": {
            "ArrayList→Vec": 0,
            "HashMap freq": 1,
            "HashSet→HashSet": 2,
            "PriorityQueue→BinaryHeap": 3,
            "Queue/Deque→VecDeque": 4,
            "Sorting": 5,
            "argmax": 6,
        },
        "mut_ownership": {
            "let mut": 0,
            "borrow conflict": 1,
            "&/&mut": 2,
            "bytes": 3,
            "split filter": 4,
            "digit→idx": 5,
            "clone()": 6,
            "in-place": 7,
        },
    }

    def __init__(self, language_path: str, java_code: str, translation_rules: Dict, merge_top_k: int = 2):
        self.language = Language(language_path, "java")
        self.parser = Parser()
        self.parser.set_language(self.language)
        self.merge_top_k = merge_top_k

        self.code_str = java_code
        self.code = java_code.encode("utf-8", errors="ignore")
        self.tree = self.parser.parse(self.code)
        self.root_node = self.tree.root_node

        self.guidance = translation_rules
        self._node_text_cache: Dict[str, str] = {}

        # rule_matches collected in one walk
        self._rule_matches_built = False
        self._rule_matches: Dict[str, int] = {c: 0 for c in self.CATS}

    # ---------- helpers ----------

    def _get_node_text(self, node: Node) -> str:
        k = f"{node.start_byte}:{node.end_byte}"
        if k in self._node_text_cache:
            return self._node_text_cache[k]
        txt = self.code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        self._node_text_cache[k] = txt
        return txt

    def _snip(self, s: str, n: int = 140) -> str:
        s = " ".join(s.split())
        return (s[:n] + "...") if len(s) > n else s

    def _is_comment_or_string(self, node: Node) -> bool:
        return node.type in ("line_comment", "block_comment", "string_literal", "char_literal")

    def _walk_all(self, visitor: Callable[[Node], None]) -> None:
        def walk(n: Node):
            visitor(n)
            for c in n.children:
                walk(c)
        walk(self.root_node)

    def _reason(self, cat: str, key: str) -> str:
        rm = self.guidance.get(cat, {}).get("translation_rule_map", {})
        return rm.get(key, "")

    def _method_name(self, node: Node) -> Optional[str]:
        # for method_invocation: name field is a node containing identifier
        n = node.child_by_field_name("name")
        if not n:
            return None
        return self._get_node_text(n)

    def _type_text(self, node: Node) -> str:
        t = node.child_by_field_name("type")
        return self._get_node_text(t) if t else self._get_node_text(node)

    def _contains_word(self, text: str, word: str) -> bool:
        # exact word boundary match
        return re.search(rf"\b{re.escape(word)}\b", text) is not None

    # NEW: distinguish real bitwise ops from boolean && and ||
    def _has_bitwise_ops(self, txt: str) -> bool:
        """
        True for real bitwise ops (<<, >>, ^, single &, single |).
        False for boolean operators && and ||.
        """
        if "<<" in txt or ">>" in txt or "^" in txt:
            return True
        if re.search(r"(?<!&)&(?!&)", txt) is not None:
            return True
        if re.search(r"(?<!\|)\|(?!\|)", txt) is not None:
            return True
        return False

    # ---------- rule_matches (single pass) ----------

    def _build_rule_matches(self) -> None:
        if self._rule_matches_built:
            return

        # reset
        self._rule_matches = {c: 0 for c in self.CATS}

        # extra counters for more precise scoring
        io_reads = 0
        io_types_used = 0
        io_custom_scanner = 0

        arr_access = 0
        arr_create = 0
        arr_2d = 0
        arr_nplus1 = 0
        arr_index_arith = 0

        num_mod = 0
        num_bit = 0
        num_mulmod = 0
        num_mod_const = 0

        ds_vec = 0
        ds_map = 0
        ds_set = 0
        ds_pq = 0
        ds_queue = 0
        ds_sort = 0
        ds_argmax = 0

        mut_charat = 0
        mut_split = 0
        mut_digit_idx = 0
        mut_borrow_conflict = 0

        def visit(n: Node):
            nonlocal io_reads, io_types_used, io_custom_scanner
            nonlocal arr_access, arr_create, arr_2d, arr_nplus1, arr_index_arith
            nonlocal num_mod, num_bit, num_mulmod, num_mod_const
            nonlocal ds_vec, ds_map, ds_set, ds_pq, ds_queue, ds_sort, ds_argmax
            nonlocal mut_charat, mut_split, mut_digit_idx, mut_borrow_conflict

            if self._is_comment_or_string(n):
                return

            # ---- IO ----
            if n.type == "method_invocation":
                mn = self._method_name(n)
                if mn in self.IO_METHODS:
                    io_reads += 1
                txt = self._get_node_text(n)
                # StringTokenizer usage often appears as new StringTokenizer(...) then nextToken()
                if "nextToken" in txt:
                    io_reads += 1

                # outputs (weak translation_rules)
                if "System.out.print" in txt or "System.out.println" in txt:
                    io_types_used += 0  # don't score on output alone

            if n.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
                t = self._get_node_text(n)
                for tp in self.IO_TYPES:
                    if self._contains_word(t, tp):
                        io_types_used += 1
                        break

                # detect custom scanner-ish classes by pattern names (strong)
                if any(self._contains_word(t, x) for x in ("FastScanner", "InputReader", "Scan")):
                    io_custom_scanner += 1

            # ---- Arrays/indexing ----
            if n.type == "array_creation_expression":
                arr_create += 1
                txt = self._get_node_text(n).replace(" ", "")
                if "][" in txt:
                    arr_2d += 1
                if "+1]" in txt:
                    arr_nplus1 += 1

            if n.type == "array_access":
                arr_access += 1
                idx = n.child_by_field_name("index")
                if idx:
                    itxt = self._get_node_text(idx).replace(" ", "")
                    if any(op in itxt for op in ("+1", "-1", "+", "-", "*", "/")):
                        arr_index_arith += 1

            # ---- Numeric/mod/bitmask ----
            if n.type == "binary_expression":
                txt = self._get_node_text(n).replace(" ", "")
                if "%" in txt:
                    num_mod += 1
                    # mulmod heuristic: contains * and % in same expr
                    if "*" in txt and txt.find("*") < txt.find("%"):
                        num_mulmod += 1
                # UPDATED: avoid counting && / || as bitwise ops
                if self._has_bitwise_ops(txt):
                    num_bit += 1

            if n.type in ("decimal_integer_literal", "integer_literal", "floating_point_literal"):
                lit = self._get_node_text(n).replace("_", "")
                if lit in self.MOD_HINTS:
                    num_mod_const += 1

            # ---- Datastructures ----
            if n.type in ("type_identifier", "scoped_type_identifier", "generic_type", "object_creation_expression"):
                t = self._get_node_text(n)

                if self._contains_word(t, "ArrayList") or self._contains_word(t, "List"):
                    ds_vec += 1
                if self._contains_word(t, "HashMap") or self._contains_word(t, "Map") or self._contains_word(t, "LinkedHashMap"):
                    ds_map += 1
                    if self._contains_word(t, "LinkedHashMap"):
                        ds_argmax += 1
                if self._contains_word(t, "HashSet") or self._contains_word(t, "Set"):
                    ds_set += 1

                if self._contains_word(t, "PriorityQueue"):
                    ds_pq += 1

                # Queue/Deque should NOT fire on "PriorityQueue"
                if (self._contains_word(t, "ArrayDeque") or self._contains_word(t, "Deque") or self._contains_word(t, "Queue")) and not self._contains_word(t, "PriorityQueue"):
                    ds_queue += 1

            if n.type == "method_invocation":
                txt = self._get_node_text(n)
                if "Arrays.sort" in txt or "Collections.sort" in txt or ".sort(" in txt:
                    ds_sort += 1
                # argmax pattern: map.put(sum,"label") or muscle.get(max)
                if ("put(" in txt and ("\"chest\"" in txt or "\"biceps\"" in txt or "\"back\"" in txt)) or ("get(max" in txt):
                    ds_argmax += 1

            # ---- Mut/ownership-lite (only CP pain points) ----
            if n.type == "method_invocation":
                txt = self._get_node_text(n)
                if "charAt(" in txt:
                    mut_charat += 1
                if ".split(" in txt:
                    mut_split += 1

            if n.type == "binary_expression":
                txt = self._get_node_text(n).replace(" ", "")
                # digit -> idx: c - '0'
                if ("-'0'" in txt) or ("-\'0\'" in txt) or ("-\"0\"" in txt):
                    mut_digit_idx += 1

            # borrow-conflict heuristic: a[i] = a[i-1] + a[i+1] style
            if n.type == "assignment_expression":
                left = n.child_by_field_name("left")
                right = n.child_by_field_name("right")
                if left and right and left.type == "array_access":
                    ltxt = self._get_node_text(left)
                    base = ltxt.split("[", 1)[0].strip()
                    rtxt = self._get_node_text(right).replace(" ", "")
                    if base and (base in rtxt) and any(x in rtxt for x in ("+1", "-1")):
                        mut_borrow_conflict += 1

        self._walk_all(visit)

        # ---- scoring ----
        # IO: require actual reads; types alone not enough (templates import IO)
        io_score = 0
        if io_reads >= 1:
            io_score += 3
        if io_custom_scanner >= 1:
            io_score += 3
        if io_reads >= 3:
            io_score += 1

        # Arrays: most CP has arrays; still useful but give modest base
        arr_score = 0
        if arr_create + arr_access > 0:
            arr_score += 2
        arr_score += min(3, arr_nplus1) * 2
        arr_score += min(2, arr_2d) * 2
        arr_score += min(3, arr_index_arith)

        # Numeric: mod/bit are strong rule_matches
        num_score = 0
        num_score += min(5, num_mod) * 1
        num_score += min(5, num_bit) * 1
        num_score += min(3, num_mulmod) * 2
        num_score += min(2, num_mod_const) * 2

        # DS: strong
        ds_score = 0
        ds_score += min(4, ds_vec)
        ds_score += min(4, ds_map) * 2
        ds_score += min(4, ds_set) * 2
        ds_score += min(4, ds_pq) * 3
        ds_score += min(4, ds_queue) * 2
        ds_score += min(3, ds_sort) * 2
        ds_score += min(2, ds_argmax) * 2

        # Mut: ONLY the CP string/digit/borrow-conflict patterns
        mut_score = 0
        mut_score += min(5, mut_charat) * 3
        mut_score += min(5, mut_split) * 3
        mut_score += min(5, mut_digit_idx) * 2
        mut_score += min(3, mut_borrow_conflict) * 3

        self._rule_matches["io_parsing"] = io_score
        self._rule_matches["arrays_indexing"] = arr_score
        self._rule_matches["numeric_mod_arith"] = num_score
        self._rule_matches["datastructures"] = ds_score
        self._rule_matches["mut_ownership"] = mut_score

        self._rule_matches_built = True

    def score_category(self) -> Dict[str, int]:
        self._build_rule_matches()
        return dict(self._rule_matches)

    def top_category(self) -> Optional[str]:
        scores = self.score_category()

        # Prefer non-IO if any non-IO category has a decent translation_rules.
        non_io = {k: v for k, v in scores.items() if k != "io_parsing"}
        best_non_io = max(non_io.items(), key=lambda kv: kv[1])
        if best_non_io[1] >= 3:
            return best_non_io[0]

        # Otherwise allow IO only if it has any translation_rules.
        if scores.get("io_parsing", 0) > 0:
            return "io_parsing"

        return None

    def detect_category_multilabel(self, threshold: int = 3) -> Dict[str, bool]:
        """
        Optional: multilabel mode (if you still want it).
        Only keep category with score >= threshold.
        """
        scores = self.score_category()
        return {c: (scores.get(c, 0) >= threshold) for c in self.CATS}

    def _dedupe_merge_block(self, block: List[Dict], cat: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Stage 1: Deduplicate by (Line, Code snippet); merge multiple hits with priority + top-k cap.
        Stage 2: Deduplicate across different lines ONLY if (Code snippet, Suggested_rust_type, Reason) is identical.
        Output: keeps only "Lines" (list), removes single "Line".
        """
        if not block:
            return []

        k = top_k if top_k is not None else self.merge_top_k
        pr = self.RUSTTYPE_PRIORITY.get(cat, {})

        groups: Dict[Tuple[int, str], List[Tuple[str, str]]] = {}
        order: List[Tuple[int, str]] = []

        def _norm_pipe(s: str) -> str:
            parts = [x.strip() for x in str(s).split("|")]
            parts = [x for x in parts if x]
            return " | ".join(parts)

        def _norm_snip(s: str) -> str:
            return " ".join(str(s).split())

        for item in block:
            line = item.get("Line")
            snippet = item.get("Code snippet", "")
            key = (line, snippet)

            if key not in groups:
                groups[key] = []
                order.append(key)

            rt_raw = str(item.get("Suggested_rust_type", "")).strip()
            rs_raw = str(item.get("Reason", "")).strip()

            rts = [x.strip() for x in rt_raw.split("|")] if rt_raw else []
            rss = [x.strip() for x in rs_raw.split("|")] if rs_raw else []

            if not rts:
                continue
            if len(rss) == 1 and len(rts) > 1:
                rss = rss * len(rts)

            for rt, reason in zip(rts, rss if rss else [""] * len(rts)):
                if rt:
                    groups[key].append((rt.strip(), reason.strip()))

        # -------------------------
        # Stage 1: merge within same (Line, snippet)
        # -------------------------
        out: List[Dict] = []
        for (line, snippet) in order:
            pairs = groups[(line, snippet)]

            uniq: Dict[str, str] = {}
            for rt, reason in pairs:
                if rt and rt not in uniq:
                    uniq[rt] = reason

            items = list(uniq.items())
            items.sort(key=lambda x: pr.get(x[0], 999))
            items = items[:k]

            out.append({
                "Line": line,
                "Code snippet": snippet,
                "Suggested_rust_type": " | ".join(rt for rt, _ in items),
                "Reason": " | ".join(reason for _, reason in items),
            })

        # -------------------------
        # Stage 2: dedupe across different lines by (snippet + final hint)
        # -------------------------
        merged: List[Dict] = []
        idx_by_key: Dict[Tuple[str, str, str], int] = {}

        for item in out:
            snippet_n = _norm_snip(item.get("Code snippet", ""))
            rt_n = _norm_pipe(item.get("Suggested_rust_type", ""))
            rs_n = _norm_pipe(item.get("Reason", ""))

            dedupe_key = (snippet_n, rt_n, rs_n)

            if dedupe_key not in idx_by_key:
                new_item = dict(item)
                ln = new_item.pop("Line", None)  
                new_item["Code snippet"] = snippet_n
                new_item["Suggested_rust_type"] = rt_n
                new_item["Reason"] = rs_n
                new_item["Lines"] = [ln] if ln is not None else []
                idx_by_key[dedupe_key] = len(merged)
                merged.append(new_item)
            else:
                m = merged[idx_by_key[dedupe_key]]
                ln = item.get("Line")
                if ln is not None and ln not in m["Lines"]:
                    m["Lines"].append(ln)

        for m in merged:
            m["Lines"] = sorted(m.get("Lines", []))

        return merged

    # ---------- translation_rules GENERATION (same schema as before) ----------

    def generate_all_rule_matches(self) -> Dict[str, List[Dict]]:
        """
        Generate rule_matches ONLY for category that score >= 3.
        Keeps translation_rules ordering stable.
        """
        cats = self.detect_category_multilabel(threshold=3)
        rule_matches: Dict[str, List[Dict]] = {}
        ref = 1

        def add(cat: str, block: List[Dict]):
            nonlocal ref
            block = self._dedupe_merge_block(block, cat)  # <-- dedupe + priority + top-k
            if block:
                rule_matches[f"translation_rules{ref}"] = block
                ref += 1

        if cats.get("io_parsing"):
            add("io_parsing", self.gen_io_parsing_rule_matches())
        if cats.get("arrays_indexing"):
            add("arrays_indexing", self.gen_arrays_indexing_rule_matches())
        if cats.get("numeric_mod_arith"):
            add("numeric_mod_arith", self.gen_numeric_mod_arith_rule_matches())
        if cats.get("datastructures"):
            add("datastructures", self.gen_datastructures_rule_matches())
        if cats.get("mut_ownership"):
            add("mut_ownership", self.gen_mut_ownership_rule_matches())

        return rule_matches

    def gen_io_parsing_rule_matches(self) -> List[Dict]:
        cat = "io_parsing"
        rm = self.guidance.get(cat, {}).get("translation_rule_map", {})
        if not rm:
            return []
        out: List[Dict] = []
        seen: Set[Tuple[int, str]] = set()

        def add(node: Node, key: str):
            if key not in rm:
                return
            line = node.start_point[0] + 1
            k = (line, key)
            if k in seen:
                return
            seen.add(k)
            out.append({
                "Line": line,
                "Code snippet": self._snip(self._get_node_text(node)),
                "Suggested_rust_type": key,
                "Reason": self._reason(cat, key),
            })

        def visit(n: Node):
            if self._is_comment_or_string(n):
                return

            if n.type == "method_invocation":
                mn = self._method_name(n)
                if mn in {"nextInt", "nextLong", "next", "readLine", "scanLong", "readInt"}:
                    add(n, "Read+split_whitespace")
                    if mn in {"nextLong", "scanLong"}:
                        add(n, "i64 then cast")

            # custom scanner class bodies: byte[] buf + read()
            if n.type == "class_declaration":
                txt = self._get_node_text(n)
                if ("byte[]" in txt or "byte []" in txt) and "InputStream" in txt and ("read(" in txt):
                    add(n, "Byte scanner")

            # printing strategy
            if n.type == "type_identifier":
                if self._contains_word(self._get_node_text(n), "PrintWriter"):
                    add(n, "String out buffer")

        self._walk_all(visit)
        return out

    def gen_arrays_indexing_rule_matches(self) -> List[Dict]:
        cat = "arrays_indexing"
        rm = self.guidance.get(cat, {}).get("translation_rule_map", {})
        if not rm:
            return []
        out: List[Dict] = []
        seen: Set[Tuple[int, str]] = set()

        def add(node: Node, key: str):
            if key not in rm:
                return
            line = node.start_point[0] + 1
            k = (line, key)
            if k in seen:
                return
            seen.add(k)
            out.append({
                "Line": line,
                "Code snippet": self._snip(self._get_node_text(node)),
                "Suggested_rust_type": key,
                "Reason": self._reason(cat, key),
            })

        def visit(n: Node):
            if self._is_comment_or_string(n):
                return

            if n.type == "array_creation_expression":
                txt = self._get_node_text(n).replace(" ", "")
                if "+1]" in txt:
                    add(n, "n+1 sizing")
                if "][" in txt:
                    add(n, "2D mapping")
                add(n, "usize index")

            if n.type == "array_access":
                add(n, "usize index")
                idx = n.child_by_field_name("index")
                if idx:
                    itxt = self._get_node_text(idx).replace(" ", "")
                    if any(x in itxt for x in ("+1", "-1", "+", "-")):
                        add(n, "bounds/offsets")

        self._walk_all(visit)
        return out

    def gen_numeric_mod_arith_rule_matches(self) -> List[Dict]:
        cat = "numeric_mod_arith"
        rm = self.guidance.get(cat, {}).get("translation_rule_map", {})
        if not rm:
            return []
        out: List[Dict] = []
        seen: Set[Tuple[int, str]] = set()

        def add(node: Node, key: str):
            if key not in rm:
                return
            line = node.start_point[0] + 1
            k = (line, key)
            if k in seen:
                return
            seen.add(k)
            out.append({
                "Line": line,
                "Code snippet": self._snip(self._get_node_text(node)),
                "Suggested_rust_type": key,
                "Reason": self._reason(cat, key),
            })

        def visit(n: Node):
            if self._is_comment_or_string(n):
                return

            if n.type == "binary_expression":
                txt = self._get_node_text(n).replace(" ", "")
                if "%" in txt:
                    add(n, "norm mod")
                    add(n, "i64 default")
                    if "*" in txt and txt.find("*") < txt.find("%"):
                        add(n, "i128 mul")
                # UPDATED: avoid && / || triggering bitmask typed
                if self._has_bitwise_ops(txt):
                    add(n, "bitmask typed")

            if n.type in ("decimal_integer_literal", "integer_literal", "floating_point_literal"):
                lit = self._get_node_text(n).replace("_", "")
                if lit in self.MOD_HINTS:
                    add(n, "norm mod")
                    add(n, "i64 default")

        self._walk_all(visit)
        return out

    def gen_datastructures_rule_matches(self) -> List[Dict]:
        cat = "datastructures"
        rm = self.guidance.get(cat, {}).get("translation_rule_map", {})
        if not rm:
            return []
        out: List[Dict] = []
        seen: Set[Tuple[int, str]] = set()

        def add(node: Node, key: str):
            if key not in rm:
                return
            line = node.start_point[0] + 1
            k = (line, key)
            if k in seen:
                return
            seen.add(k)
            out.append({
                "Line": line,
                "Code snippet": self._snip(self._get_node_text(node)),
                "Suggested_rust_type": key,
                "Reason": self._reason(cat, key),
            })

        def visit(n: Node):
            if self._is_comment_or_string(n):
                return

            if n.type in ("type_identifier", "scoped_type_identifier", "generic_type", "object_creation_expression"):
                t = self._get_node_text(n)

                if self._contains_word(t, "ArrayList") or self._contains_word(t, "List"):
                    add(n, "ArrayList→Vec")
                if self._contains_word(t, "HashMap") or self._contains_word(t, "Map") or self._contains_word(t, "LinkedHashMap"):
                    add(n, "HashMap freq")
                    if self._contains_word(t, "LinkedHashMap"):
                        add(n, "argmax")
                if self._contains_word(t, "HashSet") or self._contains_word(t, "Set"):
                    add(n, "HashSet→HashSet")

                if self._contains_word(t, "PriorityQueue"):
                    add(n, "PriorityQueue→BinaryHeap")

                if (self._contains_word(t, "ArrayDeque") or self._contains_word(t, "Deque") or self._contains_word(t, "Queue")) and not self._contains_word(t, "PriorityQueue"):
                    add(n, "Queue/Deque→VecDeque")

            if n.type == "method_invocation":
                txt = self._get_node_text(n)
                if "Arrays.sort" in txt or "Collections.sort" in txt or ".sort(" in txt:
                    add(n, "Sorting")

        self._walk_all(visit)
        return out

    def gen_mut_ownership_rule_matches(self) -> List[Dict]:
        cat = "mut_ownership"
        rm = self.guidance.get(cat, {}).get("translation_rule_map", {})
        if not rm:
            return []
        out: List[Dict] = []
        seen: Set[Tuple[int, str]] = set()

        def add(node: Node, key: str):
            if key not in rm:
                return
            line = node.start_point[0] + 1
            k = (line, key)
            if k in seen:
                return
            seen.add(k)
            out.append({
                "Line": line,
                "Code snippet": self._snip(self._get_node_text(node)),
                "Suggested_rust_type": key,
                "Reason": self._reason(cat, key),
            })

        def visit(n: Node):
            if self._is_comment_or_string(n):
                return

            if n.type == "method_invocation":
                txt = self._get_node_text(n)
                if "charAt(" in txt:
                    add(n, "bytes")
                if ".split(" in txt:
                    add(n, "split filter")

            if n.type == "binary_expression":
                txt = self._get_node_text(n).replace(" ", "")
                if ("-'0'" in txt) or ("-\'0\'" in txt):
                    add(n, "digit→idx")

            if n.type == "assignment_expression":
                left = n.child_by_field_name("left")
                right = n.child_by_field_name("right")
                if left and right and left.type == "array_access":
                    # borrow conflict hint
                    ltxt = self._get_node_text(left)
                    base = ltxt.split("[", 1)[0].strip()
                    rtxt = self._get_node_text(right).replace(" ", "")
                    if base and (base in rtxt) and any(x in rtxt for x in ("+1", "-1")):
                        add(n, "borrow conflict")

        self._walk_all(visit)
        return out


# ========== CORPUS PROCESSOR ==========

class DemonstrationCorpusProcessor:
    """
    CP corpus categorization:
      - TOP-1 category by score (default; makes corpora useful)
      - Optional multilabel mode
    """

    def __init__(self, language_path: str, RULES_PATH: str, mode: str = "top1", threshold: int = 3):
        """
        mode: "top1" or "multilabel"
        threshold: multilabel inclusion threshold
        """
        self.language_path = language_path
        with open(RULES_PATH, "r") as f:
            self.guidance = json.load(f)

        self.category = list(self.guidance.keys())  # should be your 5 CP cats
        self.mode = mode
        self.threshold = threshold

        self.counters = {c: 0 for c in self.category}
        self.counters.update({"uncategorized": 0, "total": 0, "errors": 0})

    def process_corpus(self, input_path: str, output_dir: str):
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        output_files = {c: jsonlines.open(outdir / f"{c}.jsonl", "w") for c in self.category}
        uncategorized_file = jsonlines.open(outdir / "uncategorized.jsonl", "w")

        print(f"Processing: {input_path}")
        print(f"Mode: {self.mode} (threshold={self.threshold})")
        print(f"category: {self.category}")

        with jsonlines.open(input_path, "r") as reader:
            for i, entry in enumerate(reader):
                try:
                    if "Java" not in entry or "Rust" not in entry:
                        continue

                    java_code = entry["Java"]
                    if not java_code or not java_code.strip():
                        continue

                    analyzer = JavaASTAnalyzer(self.language_path, java_code, self.guidance)

                    if self.mode == "top1":
                        cat = analyzer.top_category()
                        if cat and cat in output_files:
                            output_files[cat].write({"Java": entry["Java"], "Rust": entry["Rust"]})
                            self.counters[cat] += 1
                        else:
                            uncategorized_file.write({"Java": entry["Java"], "Rust": entry["Rust"]})
                            self.counters["uncategorized"] += 1

                    else:  # multilabel
                        cats = analyzer.detect_category_multilabel(threshold=self.threshold)
                        matched = False
                        for c, ok in cats.items():
                            if ok and c in output_files:
                                output_files[c].write({"Java": entry["Java"], "Rust": entry["Rust"]})
                                self.counters[c] += 1
                                matched = True
                        if not matched:
                            uncategorized_file.write({"Java": entry["Java"], "Rust": entry["Rust"]})
                            self.counters["uncategorized"] += 1

                    self.counters["total"] += 1

                    if (i + 1) % 500 == 0:
                        print(f"Processed {i+1}")

                except Exception as e:
                    self.counters["errors"] += 1
                    if self.counters["errors"] <= 10:
                        print(f"Error entry {i}: {str(e)[:200]}")
                    continue

        for f in output_files.values():
            f.close()
        uncategorized_file.close()

        self._print_stats(outdir)

    def _print_stats(self, outdir: Path):
        print("\n" + "=" * 60)
        print("CP CORPUS CATEGORIZATION COMPLETE")
        print("=" * 60)
        print(f"Total entries processed: {self.counters['total']}")
        print(f"Errors: {self.counters['errors']}")
        total = self.counters["total"] or 1

        print("\nEntries per category:")
        for c in sorted(self.category, key=lambda x: self.counters[x], reverse=True):
            print(f"  {c:18} {self.counters[c]:7} ({self.counters[c]/total*100:5.1f}%)")
        print(f"  {'uncategorized':18} {self.counters['uncategorized']:7} ({self.counters['uncategorized']/total*100:5.1f}%)")

        print("\nCreated files:")
        for fp in sorted(outdir.glob("*.jsonl")):
            lc = sum(1 for _ in open(fp, "r", encoding="utf-8", errors="ignore"))
            print(f"  {fp.name}: {lc} lines")


# ========== MAIN ==========

if __name__ == "__main__":
    JAVA_LANG_PATH = "/data/hassan/tree-sitter-langs/build/java-languages.so"
    RULES_PATH = "/data/hassan/java2rust-automated-migration/my-algo/translation_rules.json"
    CORPUS_PATH = "/data/hassan/java2rust-automated-migration/data/retrieval/processed/ragcorpus.jsonl"
    OUTPUT_DIR = "/data/hassan/java2rust-automated-migration/data/retrieval/categories"

    MODE = "top1"       # "top1" or "multilabel"
    THRESHOLD = 3       # used only if MODE="multilabel"

    print("Starting CP Java Corpus Categorization")
    print("=" * 60)
    print(f"guidance: {RULES_PATH}")
    print(f"Corpus: {CORPUS_PATH}")
    print(f"Out:   {OUTPUT_DIR}")
    print(f"Mode:  {MODE}")
    print("=" * 60)

    proc = DemonstrationCorpusProcessor(JAVA_LANG_PATH, RULES_PATH, mode=MODE, threshold=THRESHOLD)
    proc.process_corpus(CORPUS_PATH, OUTPUT_DIR)

    # sanity test
    test_java = r"""
    import java.util.*;
    class Main{
      public static void main(String[] args){
        Scanner sc=new Scanner(System.in);
        long n=sc.nextLong();
        long mask=0;
        mask |= (1<<3);
        String s=sc.next();
        if(s.charAt(0)=='X') System.out.println("YES");
        int[] a=new int[(int)n+1];
        Arrays.sort(a);
        PriorityQueue<Integer> pq=new PriorityQueue<>();
      }
    }
    """
    analyzer = JavaASTAnalyzer(JAVA_LANG_PATH, test_java, proc.guidance)
    print("\nScores:", analyzer.score_category())
    print("Top category:", analyzer.top_category())
    print("rule_matches:", json.dumps(analyzer.generate_all_rule_matches(), indent=2))
