from __future__ import annotations

import ast
import argparse
import copy
import hashlib
import itertools
import json
import random
import re
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_chapter2_capability import (  # noqa: E402
    build_messages,
    extract_gsm8k_reference,
    load_cmmlu_sample,
    load_gsm8k_sample,
    load_humaneval_sample,
    read_jsonl,
    read_split_ids,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT = ROOT / "data" / "distill" / "capability_rehearsal_v3.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "audit" / "gate_kd_capability_rehearsal_v3.json"


CODE_TASKS = [
    (
        "synthetic_code/add_abs",
        "def add_abs(a: int, b: int) -> int:\n    \"\"\"Return abs(a) + abs(b).\"\"\"\n",
        "    return abs(a) + abs(b)",
    ),
    (
        "synthetic_code/count_vowels",
        "def count_vowels(text: str) -> int:\n    \"\"\"Return the number of vowels in text.\"\"\"\n",
        "    return sum(1 for ch in text.lower() if ch in 'aeiou')",
    ),
    (
        "synthetic_code/is_palindrome",
        "def is_palindrome(text: str) -> bool:\n    \"\"\"Return True if text is a palindrome after removing spaces.\"\"\"\n",
        "    cleaned = ''.join(text.lower().split())\n    return cleaned == cleaned[::-1]",
    ),
    (
        "synthetic_code/flatten_once",
        "def flatten_once(items: list[list[int]]) -> list[int]:\n    \"\"\"Flatten a list of integer lists by one level.\"\"\"\n",
        "    return [value for group in items for value in group]",
    ),
    (
        "synthetic_code/factorial",
        "def factorial(n: int) -> int:\n    \"\"\"Return n factorial for n >= 0.\"\"\"\n",
        "    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result",
    ),
    (
        "synthetic_code/unique_sorted",
        "def unique_sorted(values: list[int]) -> list[int]:\n    \"\"\"Return sorted unique integers.\"\"\"\n",
        "    return sorted(set(values))",
    ),
    (
        "synthetic_code/merge_dict_counts",
        "def merge_dict_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:\n    \"\"\"Merge count dictionaries by summing values.\"\"\"\n",
        "    result = dict(a)\n    for key, value in b.items():\n        result[key] = result.get(key, 0) + value\n    return result",
    ),
    (
        "synthetic_code/second_largest",
        "def second_largest(values: list[int]) -> int | None:\n    \"\"\"Return the second largest distinct value, or None.\"\"\"\n",
        "    unique = sorted(set(values))\n    return unique[-2] if len(unique) >= 2 else None",
    ),
]

CODE_TASKS.extend(
    [
        (
            "synthetic_code/running_total",
            "def running_total(values: list[int]) -> list[int]:\n    \"\"\"Return prefix sums for values.\"\"\"\n",
            "    total = 0\n    result = []\n    for value in values:\n        total += value\n        result.append(total)\n    return result",
        ),
        (
            "synthetic_code/filter_by_prefix",
            "def filter_by_prefix(words: list[str], prefix: str) -> list[str]:\n    \"\"\"Return words that start with prefix.\"\"\"\n",
            "    return [word for word in words if word.startswith(prefix)]",
        ),
        (
            "synthetic_code/rotate_left",
            "def rotate_left(values: list[int], k: int) -> list[int]:\n    \"\"\"Rotate values left by k positions.\"\"\"\n",
            "    if not values:\n        return []\n    k %= len(values)\n    return values[k:] + values[:k]",
        ),
        (
            "synthetic_code/is_prime",
            "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime.\"\"\"\n",
            "    if n < 2:\n        return False\n    for value in range(2, int(n ** 0.5) + 1):\n        if n % value == 0:\n            return False\n    return True",
        ),
        (
            "synthetic_code/group_by_length",
            "def group_by_length(words: list[str]) -> dict[int, list[str]]:\n    \"\"\"Group words by their length.\"\"\"\n",
            "    result = {}\n    for word in words:\n        result.setdefault(len(word), []).append(word)\n    return result",
        ),
        (
            "synthetic_code/remove_duplicates_keep_order",
            "def remove_duplicates_keep_order(values: list[int]) -> list[int]:\n    \"\"\"Remove duplicates while preserving first occurrence.\"\"\"\n",
            "    seen = set()\n    result = []\n    for value in values:\n        if value not in seen:\n            seen.add(value)\n            result.append(value)\n    return result",
        ),
        (
            "synthetic_code/sum_matrix_diagonal",
            "def sum_matrix_diagonal(matrix: list[list[int]]) -> int:\n    \"\"\"Return the main diagonal sum.\"\"\"\n",
            "    return sum(row[i] for i, row in enumerate(matrix) if i < len(row))",
        ),
        (
            "synthetic_code/transpose",
            "def transpose(matrix: list[list[int]]) -> list[list[int]]:\n    \"\"\"Transpose a rectangular matrix.\"\"\"\n",
            "    if not matrix:\n        return []\n    return [list(row) for row in zip(*matrix)]",
        ),
        (
            "synthetic_code/find_first_even",
            "def find_first_even(values: list[int]) -> int | None:\n    \"\"\"Return the first even number, or None.\"\"\"\n",
            "    for value in values:\n        if value % 2 == 0:\n            return value\n    return None",
        ),
        (
            "synthetic_code/clamp",
            "def clamp(value: int, low: int, high: int) -> int:\n    \"\"\"Clamp value into [low, high].\"\"\"\n",
            "    return max(low, min(high, value))",
        ),
        (
            "synthetic_code/count_words",
            "def count_words(text: str) -> dict[str, int]:\n    \"\"\"Count whitespace-separated words.\"\"\"\n",
            "    result = {}\n    for word in text.split():\n        result[word] = result.get(word, 0) + 1\n    return result",
        ),
        (
            "synthetic_code/reverse_words",
            "def reverse_words(text: str) -> str:\n    \"\"\"Reverse word order in text.\"\"\"\n",
            "    return ' '.join(reversed(text.split()))",
        ),
        (
            "synthetic_code/common_items",
            "def common_items(a: list[int], b: list[int]) -> list[int]:\n    \"\"\"Return sorted unique values present in both lists.\"\"\"\n",
            "    return sorted(set(a) & set(b))",
        ),
        (
            "synthetic_code/chunk",
            "def chunk(values: list[int], size: int) -> list[list[int]]:\n    \"\"\"Split values into chunks of size.\"\"\"\n",
            "    return [values[i:i + size] for i in range(0, len(values), size)]",
        ),
        (
            "synthetic_code/balanced_parentheses",
            "def balanced_parentheses(text: str) -> bool:\n    \"\"\"Return True if parentheses are balanced.\"\"\"\n",
            "    depth = 0\n    for ch in text:\n        if ch == '(':\n            depth += 1\n        elif ch == ')':\n            depth -= 1\n            if depth < 0:\n                return False\n    return depth == 0",
        ),
        (
            "synthetic_code/longest_word",
            "def longest_word(words: list[str]) -> str:\n    \"\"\"Return the longest word, or empty string.\"\"\"\n",
            "    return max(words, key=len) if words else ''",
        ),
        (
            "synthetic_code/digits_sum",
            "def digits_sum(n: int) -> int:\n    \"\"\"Return the sum of decimal digits of abs(n).\"\"\"\n",
            "    return sum(int(ch) for ch in str(abs(n)))",
        ),
        (
            "synthetic_code/pair_sums",
            "def pair_sums(values: list[int]) -> list[int]:\n    \"\"\"Return sums of adjacent pairs.\"\"\"\n",
            "    return [values[i] + values[i + 1] for i in range(len(values) - 1)]",
        ),
        (
            "synthetic_code/title_case",
            "def title_case(text: str) -> str:\n    \"\"\"Capitalize every word.\"\"\"\n",
            "    return ' '.join(word.capitalize() for word in text.split())",
        ),
        (
            "synthetic_code/all_positive",
            "def all_positive(values: list[int]) -> bool:\n    \"\"\"Return True if every value is positive.\"\"\"\n",
            "    return all(value > 0 for value in values)",
        ),
        (
            "synthetic_code/window_max",
            "def window_max(values: list[int], size: int) -> list[int]:\n    \"\"\"Return max for each sliding window.\"\"\"\n",
            "    return [max(values[i:i + size]) for i in range(0, len(values) - size + 1)]",
        ),
        (
            "synthetic_code/flatten_dict_values",
            "def flatten_dict_values(data: dict[str, list[int]]) -> list[int]:\n    \"\"\"Flatten all list values in key order.\"\"\"\n",
            "    result = []\n    for key in sorted(data):\n        result.extend(data[key])\n    return result",
        ),
        (
            "synthetic_code/has_duplicate",
            "def has_duplicate(values: list[int]) -> bool:\n    \"\"\"Return True if any value repeats.\"\"\"\n",
            "    return len(set(values)) != len(values)",
        ),
        (
            "synthetic_code/median_sorted",
            "def median_sorted(values: list[int]) -> float:\n    \"\"\"Return median after sorting a non-empty list.\"\"\"\n",
            "    ordered = sorted(values)\n    mid = len(ordered) // 2\n    if len(ordered) % 2 == 1:\n        return float(ordered[mid])\n    return (ordered[mid - 1] + ordered[mid]) / 2",
        ),
    ]
)

CODE_TASKS.extend(
    [
        (
            "synthetic_code/sort_by_second_then_first",
            "def sort_by_second_then_first(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    \"\"\"Sort pairs by second value, then first value.\"\"\"\n",
            "    return sorted(pairs, key=lambda item: (item[1], item[0]))",
        ),
        (
            "synthetic_code/invert_dict",
            "def invert_dict(data: dict[str, int]) -> dict[int, list[str]]:\n    \"\"\"Group keys by their integer values, preserving sorted key order.\"\"\"\n",
            "    result = {}\n    for key in sorted(data):\n        result.setdefault(data[key], []).append(key)\n    return result",
        ),
        (
            "synthetic_code/longest_increasing_run",
            "def longest_increasing_run(values: list[int]) -> int:\n    \"\"\"Return the length of the longest strictly increasing contiguous run.\"\"\"\n",
            "    if not values:\n        return 0\n    best = current = 1\n    for i in range(1, len(values)):\n        if values[i] > values[i - 1]:\n            current += 1\n        else:\n            current = 1\n        best = max(best, current)\n    return best",
        ),
        (
            "synthetic_code/normalize_spaces",
            "def normalize_spaces(text: str) -> str:\n    \"\"\"Collapse whitespace and strip the result.\"\"\"\n",
            "    return ' '.join(text.split())",
        ),
        (
            "synthetic_code/frequency_order",
            "def frequency_order(values: list[str]) -> list[str]:\n    \"\"\"Return unique strings sorted by decreasing frequency then alphabetically.\"\"\"\n",
            "    counts = {}\n    for value in values:\n        counts[value] = counts.get(value, 0) + 1\n    return sorted(counts, key=lambda value: (-counts[value], value))",
        ),
        (
            "synthetic_code/rle_encode",
            "def rle_encode(text: str) -> list[tuple[str, int]]:\n    \"\"\"Run-length encode a string as (character, count) pairs.\"\"\"\n",
            "    if not text:\n        return []\n    result = []\n    current = text[0]\n    count = 1\n    for ch in text[1:]:\n        if ch == current:\n            count += 1\n        else:\n            result.append((current, count))\n            current = ch\n            count = 1\n    result.append((current, count))\n    return result",
        ),
        (
            "synthetic_code/safe_divide",
            "def safe_divide(a: float, b: float) -> float | None:\n    \"\"\"Return a / b, or None when b is zero.\"\"\"\n",
            "    if b == 0:\n        return None\n    return a / b",
        ),
        (
            "synthetic_code/top_k",
            "def top_k(values: list[int], k: int) -> list[int]:\n    \"\"\"Return the k largest values in descending order.\"\"\"\n",
            "    if k <= 0:\n        return []\n    return sorted(values, reverse=True)[:k]",
        ),
        (
            "synthetic_code/is_anagram",
            "def is_anagram(a: str, b: str) -> bool:\n    \"\"\"Return True if two strings are anagrams ignoring spaces and case.\"\"\"\n",
            "    clean_a = ''.join(a.lower().split())\n    clean_b = ''.join(b.lower().split())\n    return sorted(clean_a) == sorted(clean_b)",
        ),
        (
            "synthetic_code/sliding_average",
            "def sliding_average(values: list[float], size: int) -> list[float]:\n    \"\"\"Return averages of each sliding window.\"\"\"\n",
            "    if size <= 0 or size > len(values):\n        return []\n    return [sum(values[i:i + size]) / size for i in range(len(values) - size + 1)]",
        ),
    ]
)

ADVANCED_CODE_TASKS = [
    (
        "synthetic_code_advanced/merge_intervals",
        "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    \"\"\"Merge overlapping closed intervals sorted by start.\"\"\"\n",
        "    if not intervals:\n        return []\n    intervals = sorted(intervals)\n    merged = [intervals[0][:]]\n    for start, end in intervals[1:]:\n        if start <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], end)\n        else:\n            merged.append([start, end])\n    return merged",
    ),
    (
        "synthetic_code_advanced/valid_parentheses",
        "def valid_parentheses(text: str) -> bool:\n    \"\"\"Return True if (), [], and {} brackets are balanced.\"\"\"\n",
        "    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in text:\n        if ch in pairs.values():\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack",
    ),
    (
        "synthetic_code_advanced/longest_common_prefix",
        "def longest_common_prefix(words: list[str]) -> str:\n    \"\"\"Return the longest common prefix of all words.\"\"\"\n",
        "    if not words:\n        return ''\n    prefix = words[0]\n    for word in words[1:]:\n        while not word.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix",
    ),
    (
        "synthetic_code_advanced/two_sum_indices",
        "def two_sum_indices(values: list[int], target: int) -> tuple[int, int] | None:\n    \"\"\"Return indices of two values that sum to target, or None.\"\"\"\n",
        "    seen = {}\n    for index, value in enumerate(values):\n        need = target - value\n        if need in seen:\n            return (seen[need], index)\n        seen[value] = index\n    return None",
    ),
    (
        "synthetic_code_advanced/binary_search",
        "def binary_search(values: list[int], target: int) -> int:\n    \"\"\"Return target index in sorted values, or -1.\"\"\"\n",
        "    lo, hi = 0, len(values) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if values[mid] == target:\n            return mid\n        if values[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    ),
    (
        "synthetic_code_advanced/edit_distance_one",
        "def edit_distance_one(a: str, b: str) -> bool:\n    \"\"\"Return True if strings are exactly one insert, delete, or replace apart.\"\"\"\n",
        "    if abs(len(a) - len(b)) > 1:\n        return False\n    if len(a) > len(b):\n        a, b = b, a\n    i = j = edits = 0\n    while i < len(a) and j < len(b):\n        if a[i] == b[j]:\n            i += 1\n            j += 1\n        else:\n            edits += 1\n            if edits > 1:\n                return False\n            if len(a) == len(b):\n                i += 1\n            j += 1\n    edits += len(b) - j\n    return edits == 1",
    ),
    (
        "synthetic_code_advanced/topological_layers",
        "def topological_layers(edges: list[tuple[str, str]]) -> list[list[str]]:\n    \"\"\"Return nodes grouped by topological layer for a DAG.\"\"\"\n",
        "    children = {}\n    indeg = {}\n    for src, dst in edges:\n        children.setdefault(src, []).append(dst)\n        indeg.setdefault(src, 0)\n        indeg[dst] = indeg.get(dst, 0) + 1\n    layer = sorted(node for node, degree in indeg.items() if degree == 0)\n    result = []\n    while layer:\n        result.append(layer)\n        next_layer = []\n        for node in layer:\n            for child in children.get(node, []):\n                indeg[child] -= 1\n                if indeg[child] == 0:\n                    next_layer.append(child)\n        layer = sorted(next_layer)\n    return result",
    ),
    (
        "synthetic_code_advanced/roman_to_int",
        "def roman_to_int(text: str) -> int:\n    \"\"\"Convert a Roman numeral using standard subtractive notation.\"\"\"\n",
        "    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    previous = 0\n    for ch in reversed(text):\n        value = values[ch]\n        if value < previous:\n            total -= value\n        else:\n            total += value\n            previous = value\n    return total",
    ),
    (
        "synthetic_code_advanced/spiral_order",
        "def spiral_order(matrix: list[list[int]]) -> list[int]:\n    \"\"\"Return matrix values in clockwise spiral order.\"\"\"\n",
        "    result = []\n    while matrix and matrix[0]:\n        result += matrix.pop(0)\n        matrix = [list(row) for row in zip(*matrix)][::-1]\n    return result",
    ),
    (
        "synthetic_code_advanced/compress_ranges",
        "def compress_ranges(values: list[int]) -> list[str]:\n    \"\"\"Compress sorted integers into range strings.\"\"\"\n",
        "    if not values:\n        return []\n    result = []\n    start = prev = values[0]\n    for value in values[1:]:\n        if value == prev + 1:\n            prev = value\n            continue\n        result.append(str(start) if start == prev else f'{start}-{prev}')\n        start = prev = value\n    result.append(str(start) if start == prev else f'{start}-{prev}')\n    return result",
    ),
    (
        "synthetic_code_advanced/min_window_subsequence",
        "def min_window_subsequence(text: str, target: str) -> str:\n    \"\"\"Return the shortest substring of text containing target as a subsequence.\"\"\"\n",
        "    best = ''\n    for start in range(len(text)):\n        j = 0\n        for end in range(start, len(text)):\n            if text[end] == target[j]:\n                j += 1\n                if j == len(target):\n                    candidate = text[start:end + 1]\n                    if not best or len(candidate) < len(best):\n                        best = candidate\n                    break\n    return best",
    ),
    (
        "synthetic_code_advanced/tree_levels_from_edges",
        "def tree_levels_from_edges(root: str, edges: list[tuple[str, str]]) -> list[list[str]]:\n    \"\"\"Return breadth-first levels from directed parent-child edges.\"\"\"\n",
        "    children = {}\n    for parent, child in edges:\n        children.setdefault(parent, []).append(child)\n    result = []\n    level = [root]\n    while level:\n        result.append(level)\n        next_level = []\n        for node in level:\n            next_level.extend(sorted(children.get(node, [])))\n        level = next_level\n    return result",
    ),
]

ADVANCED_CODE_TASKS.extend(
    [
        (
            "synthetic_code_advanced/gcd_iterative",
            "def gcd_iterative(a: int, b: int) -> int:\n    \"\"\"Return the non-negative greatest common divisor of a and b.\"\"\"\n",
            "    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a",
        ),
        (
            "synthetic_code_advanced/lcs_length",
            "def lcs_length(a: str, b: str) -> int:\n    \"\"\"Return the length of the longest common subsequence.\"\"\"\n",
            "    previous = [0] * (len(b) + 1)\n    for left in a:\n        current = [0]\n        for index, right in enumerate(b, 1):\n            if left == right:\n                current.append(previous[index - 1] + 1)\n            else:\n                current.append(max(previous[index], current[-1]))\n        previous = current\n    return previous[-1]",
        ),
        (
            "synthetic_code_advanced/min_coins",
            "def min_coins(coins: list[int], amount: int) -> int:\n    \"\"\"Return the minimum coins needed for amount, or -1 when impossible.\"\"\"\n",
            "    best = [amount + 1] * (amount + 1)\n    best[0] = 0\n    for total in range(1, amount + 1):\n        for coin in coins:\n            if coin <= total:\n                best[total] = min(best[total], best[total - coin] + 1)\n    return best[amount] if best[amount] <= amount else -1",
        ),
        (
            "synthetic_code_advanced/shortest_path_length",
            "def shortest_path_length(edges: list[tuple[str, str]], start: str, end: str) -> int:\n    \"\"\"Return the unweighted undirected path length, or -1 when unreachable.\"\"\"\n",
            "    graph = {}\n    for left, right in edges:\n        graph.setdefault(left, []).append(right)\n        graph.setdefault(right, []).append(left)\n    queue = [(start, 0)]\n    seen = {start}\n    for node, distance in queue:\n        if node == end:\n            return distance\n        for neighbor in graph.get(node, []):\n            if neighbor not in seen:\n                seen.add(neighbor)\n                queue.append((neighbor, distance + 1))\n    return -1",
        ),
        (
            "synthetic_code_advanced/rotate_matrix_clockwise",
            "def rotate_matrix_clockwise(matrix: list[list[int]]) -> list[list[int]]:\n    \"\"\"Return a rectangular matrix rotated 90 degrees clockwise.\"\"\"\n",
            "    if not matrix:\n        return []\n    return [list(row) for row in zip(*matrix[::-1])]",
        ),
        (
            "synthetic_code_advanced/group_anagrams",
            "def group_anagrams(words: list[str]) -> list[list[str]]:\n    \"\"\"Group anagrams and return deterministically ordered groups.\"\"\"\n",
            "    groups = {}\n    for word in words:\n        key = ''.join(sorted(word))\n        groups.setdefault(key, []).append(word)\n    return [sorted(groups[key]) for key in sorted(groups)]",
        ),
        (
            "synthetic_code_advanced/rle_decode",
            "def rle_decode(items: list[tuple[str, int]]) -> str:\n    \"\"\"Decode (character, count) run-length pairs.\"\"\"\n",
            "    return ''.join(character * count for character, count in items if count > 0)",
        ),
        (
            "synthetic_code_advanced/longest_unique_substring",
            "def longest_unique_substring(text: str) -> int:\n    \"\"\"Return the longest substring length without repeated characters.\"\"\"\n",
            "    last = {}\n    start = 0\n    best = 0\n    for index, character in enumerate(text):\n        if character in last and last[character] >= start:\n            start = last[character] + 1\n        last[character] = index\n        best = max(best, index - start + 1)\n    return best",
        ),
        (
            "synthetic_code_advanced/merge_sorted_lists",
            "def merge_sorted_lists(a: list[int], b: list[int]) -> list[int]:\n    \"\"\"Merge two sorted integer lists while preserving duplicates.\"\"\"\n",
            "    result = []\n    left = right = 0\n    while left < len(a) and right < len(b):\n        if a[left] <= b[right]:\n            result.append(a[left])\n            left += 1\n        else:\n            result.append(b[right])\n            right += 1\n    return result + a[left:] + b[right:]",
        ),
        (
            "synthetic_code_advanced/pascal_row",
            "def pascal_row(index: int) -> list[int]:\n    \"\"\"Return the zero-indexed row of Pascal's triangle.\"\"\"\n",
            "    row = [1]\n    for step in range(1, index + 1):\n        row.append(row[-1] * (index - step + 1) // step)\n    return row",
        ),
        (
            "synthetic_code_advanced/evaluate_rpn",
            "def evaluate_rpn(tokens: list[str]) -> int:\n    \"\"\"Evaluate reverse Polish notation with +, -, *, and truncating division.\"\"\"\n",
            "    stack = []\n    for token in tokens:\n        if token not in {'+', '-', '*', '/'}:\n            stack.append(int(token))\n            continue\n        right = stack.pop()\n        left = stack.pop()\n        if token == '+':\n            stack.append(left + right)\n        elif token == '-':\n            stack.append(left - right)\n        elif token == '*':\n            stack.append(left * right)\n        else:\n            stack.append(int(left / right))\n    return stack[-1]",
        ),
        (
            "synthetic_code_advanced/connected_components",
            "def connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:\n    \"\"\"Return sorted connected components of an undirected graph.\"\"\"\n",
            "    graph = {node: [] for node in nodes}\n    for left, right in edges:\n        graph.setdefault(left, []).append(right)\n        graph.setdefault(right, []).append(left)\n    seen = set()\n    result = []\n    for start in sorted(graph):\n        if start in seen:\n            continue\n        stack = [start]\n        seen.add(start)\n        component = []\n        while stack:\n            node = stack.pop()\n            component.append(node)\n            for neighbor in graph[node]:\n                if neighbor not in seen:\n                    seen.add(neighbor)\n                    stack.append(neighbor)\n        result.append(sorted(component))\n    return result",
        ),
        (
            "synthetic_code_advanced/k_closest_values",
            "def k_closest_values(values: list[int], target: int, k: int) -> list[int]:\n    \"\"\"Return k values ordered by distance to target, then value.\"\"\"\n",
            "    return sorted(values, key=lambda value: (abs(value - target), value))[:max(k, 0)]",
        ),
        (
            "synthetic_code_advanced/word_ladder_neighbors",
            "def word_ladder_neighbors(word: str, candidates: list[str]) -> list[str]:\n    \"\"\"Return sorted candidates differing from word at exactly one position.\"\"\"\n",
            "    result = []\n    for candidate in candidates:\n        if len(candidate) != len(word):\n            continue\n        differences = sum(left != right for left, right in zip(word, candidate))\n        if differences == 1:\n            result.append(candidate)\n    return sorted(result)",
        ),
        (
            "synthetic_code_advanced/subarray_sum_count",
            "def subarray_sum_count(values: list[int], target: int) -> int:\n    \"\"\"Count contiguous subarrays whose sum equals target.\"\"\"\n",
            "    prefixes = {0: 1}\n    total = 0\n    result = 0\n    for value in values:\n        total += value\n        result += prefixes.get(total - target, 0)\n        prefixes[total] = prefixes.get(total, 0) + 1\n    return result",
        ),
        (
            "synthetic_code_advanced/interval_intersection",
            "def interval_intersection(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:\n    \"\"\"Return intersections of two sorted disjoint interval lists.\"\"\"\n",
            "    result = []\n    left = right = 0\n    while left < len(a) and right < len(b):\n        start = max(a[left][0], b[right][0])\n        end = min(a[left][1], b[right][1])\n        if start <= end:\n            result.append([start, end])\n        if a[left][1] < b[right][1]:\n            left += 1\n        else:\n            right += 1\n    return result",
        ),
    ]
)


# v22 adds new semantic families without changing the frozen v21 source generator.
# Keep these tasks behind dedicated CLI flags so rerunning build-rehearsal-v21 is stable.
V22_CODE_TASKS = [
    (
        "synthetic_code_v22/sum_even_values",
        "def sum_even_values(values: list[int]) -> int:\n    \"\"\"Return the sum of all even integers in values.\"\"\"\n",
        "    return sum(value for value in values if value % 2 == 0)",
    ),
    (
        "synthetic_code_v22/product_nonzero",
        "def product_nonzero(values: list[int]) -> int:\n    \"\"\"Multiply nonzero values, returning 1 when none exist.\"\"\"\n",
        "    result = 1\n    for value in values:\n        if value != 0:\n            result *= value\n    return result",
    ),
    (
        "synthetic_code_v22/count_occurrences",
        "def count_occurrences(values: list[int], target: int) -> int:\n    \"\"\"Return how many times target occurs in values.\"\"\"\n",
        "    return sum(value == target for value in values)",
    ),
    (
        "synthetic_code_v22/last_index",
        "def last_index(values: list[int], target: int) -> int:\n    \"\"\"Return the last target index, or -1 when absent.\"\"\"\n",
        "    for index in range(len(values) - 1, -1, -1):\n        if values[index] == target:\n            return index\n    return -1",
    ),
    (
        "synthetic_code_v22/min_max_span",
        "def min_max_span(values: list[int]) -> int:\n    \"\"\"Return max(values)-min(values), or 0 for an empty list.\"\"\"\n",
        "    return max(values) - min(values) if values else 0",
    ),
    (
        "synthetic_code_v22/prefix_max",
        "def prefix_max(values: list[int]) -> list[int]:\n    \"\"\"Return the maximum observed at every prefix position.\"\"\"\n",
        "    result = []\n    for value in values:\n        result.append(value if not result else max(result[-1], value))\n    return result",
    ),
    (
        "synthetic_code_v22/suffix_min",
        "def suffix_min(values: list[int]) -> list[int]:\n    \"\"\"Return the minimum observed at every suffix position.\"\"\"\n",
        "    result = []\n    current = None\n    for value in reversed(values):\n        current = value if current is None else min(current, value)\n        result.append(current)\n    return result[::-1]",
    ),
    (
        "synthetic_code_v22/adjacent_differences",
        "def adjacent_differences(values: list[int]) -> list[int]:\n    \"\"\"Return each value minus its immediate predecessor.\"\"\"\n",
        "    return [values[index] - values[index - 1] for index in range(1, len(values))]",
    ),
    (
        "synthetic_code_v22/moving_sum",
        "def moving_sum(values: list[int], width: int) -> list[int]:\n    \"\"\"Return sums of complete windows of positive width.\"\"\"\n",
        "    if width <= 0 or width > len(values):\n        return []\n    return [sum(values[index:index + width]) for index in range(len(values) - width + 1)]",
    ),
    (
        "synthetic_code_v22/rotate_right",
        "def rotate_right(values: list[int], amount: int) -> list[int]:\n    \"\"\"Rotate values right by amount, supporting negative amounts.\"\"\"\n",
        "    if not values:\n        return []\n    amount %= len(values)\n    return values[-amount:] + values[:-amount] if amount else list(values)",
    ),
    (
        "synthetic_code_v22/partition_even_odd",
        "def partition_even_odd(values: list[int]) -> list[int]:\n    \"\"\"Return evens then odds while preserving order within each part.\"\"\"\n",
        "    return [value for value in values if value % 2 == 0] + [value for value in values if value % 2 != 0]",
    ),
    (
        "synthetic_code_v22/longest_equal_run",
        "def longest_equal_run(values: list[int]) -> int:\n    \"\"\"Return the longest contiguous run of one repeated value.\"\"\"\n",
        "    best = current = 0\n    previous = None\n    for value in values:\n        current = current + 1 if current and value == previous else 1\n        best = max(best, current)\n        previous = value\n    return best",
    ),
    (
        "synthetic_code_v22/count_inversions",
        "def count_inversions(values: list[int]) -> int:\n    \"\"\"Count index pairs i<j whose values are in descending order.\"\"\"\n",
        "    return sum(values[left] > values[right] for left in range(len(values)) for right in range(left + 1, len(values)))",
    ),
    (
        "synthetic_code_v22/dot_product",
        "def dot_product(a: list[int], b: list[int]) -> int:\n    \"\"\"Return the dot product up to the shorter input length.\"\"\"\n",
        "    return sum(left * right for left, right in zip(a, b))",
    ),
    (
        "synthetic_code_v22/matrix_row_sums",
        "def matrix_row_sums(matrix: list[list[int]]) -> list[int]:\n    \"\"\"Return the sum of each matrix row.\"\"\"\n",
        "    return [sum(row) for row in matrix]",
    ),
    (
        "synthetic_code_v22/matrix_column_sums",
        "def matrix_column_sums(matrix: list[list[int]]) -> list[int]:\n    \"\"\"Return column sums for a rectangular matrix.\"\"\"\n",
        "    if not matrix or not matrix[0]:\n        return []\n    return [sum(row[column] for row in matrix) for column in range(len(matrix[0]))]",
    ),
    (
        "synthetic_code_v22/diagonal_difference",
        "def diagonal_difference(matrix: list[list[int]]) -> int:\n    \"\"\"Return main diagonal sum minus anti-diagonal sum.\"\"\"\n",
        "    if not matrix or not matrix[0]:\n        return 0\n    size = min(len(matrix), len(matrix[0]))\n    return sum(matrix[index][index] - matrix[index][size - index - 1] for index in range(size))",
    ),
    (
        "synthetic_code_v22/longest_common_suffix",
        "def longest_common_suffix(a: str, b: str) -> str:\n    \"\"\"Return the longest suffix shared by both strings.\"\"\"\n",
        "    length = 0\n    while length < min(len(a), len(b)) and a[-length - 1] == b[-length - 1]:\n        length += 1\n    return a[len(a) - length:] if length else ''",
    ),
    (
        "synthetic_code_v22/count_overlapping_substring",
        "def count_overlapping_substring(text: str, pattern: str) -> int:\n    \"\"\"Count overlapping pattern occurrences; an empty pattern counts zero.\"\"\"\n",
        "    if not pattern:\n        return 0\n    return sum(text.startswith(pattern, index) for index in range(len(text) - len(pattern) + 1))",
    ),
    (
        "synthetic_code_v22/remove_consecutive_characters",
        "def remove_consecutive_characters(text: str) -> str:\n    \"\"\"Collapse each run of equal consecutive characters to one.\"\"\"\n",
        "    result = []\n    for character in text:\n        if not result or result[-1] != character:\n            result.append(character)\n    return ''.join(result)",
    ),
    (
        "synthetic_code_v22/first_unique_character",
        "def first_unique_character(text: str) -> str:\n    \"\"\"Return the first character occurring once, or an empty string.\"\"\"\n",
        "    counts = {character: text.count(character) for character in set(text)}\n    return next((character for character in text if counts[character] == 1), '')",
    ),
    (
        "synthetic_code_v22/character_histogram",
        "def character_histogram(text: str) -> dict[str, int]:\n    \"\"\"Return character occurrence counts.\"\"\"\n",
        "    result = {}\n    for character in text:\n        result[character] = result.get(character, 0) + 1\n    return result",
    ),
    (
        "synthetic_code_v22/word_lengths",
        "def word_lengths(words: list[str]) -> list[int]:\n    \"\"\"Return the length of every word in order.\"\"\"\n",
        "    return [len(word) for word in words]",
    ),
    (
        "synthetic_code_v22/sort_words_by_length",
        "def sort_words_by_length(words: list[str]) -> list[str]:\n    \"\"\"Sort words by length and then lexicographically.\"\"\"\n",
        "    return sorted(words, key=lambda word: (len(word), word))",
    ),
    (
        "synthetic_code_v22/common_prefix_length",
        "def common_prefix_length(a: str, b: str) -> int:\n    \"\"\"Return the number of leading characters shared by a and b.\"\"\"\n",
        "    length = 0\n    while length < min(len(a), len(b)) and a[length] == b[length]:\n        length += 1\n    return length",
    ),
    (
        "synthetic_code_v22/caesar_shift_lowercase",
        "def caesar_shift_lowercase(text: str, shift: int) -> str:\n    \"\"\"Shift lowercase ASCII letters and leave other characters unchanged.\"\"\"\n",
        "    result = []\n    for character in text:\n        if 'a' <= character <= 'z':\n            result.append(chr((ord(character) - ord('a') + shift) % 26 + ord('a')))\n        else:\n            result.append(character)\n    return ''.join(result)",
    ),
    (
        "synthetic_code_v22/digital_root",
        "def digital_root(value: int) -> int:\n    \"\"\"Return the repeated digit sum of the absolute integer.\"\"\"\n",
        "    value = abs(value)\n    while value >= 10:\n        value = sum(int(digit) for digit in str(value))\n    return value",
    ),
    (
        "synthetic_code_v22/gcd_list",
        "def gcd_list(values: list[int]) -> int:\n    \"\"\"Return the non-negative greatest common divisor of all values.\"\"\"\n",
        "    result = 0\n    for value in values:\n        left, right = result, abs(value)\n        while right:\n            left, right = right, left % right\n        result = left\n    return result",
    ),
    (
        "synthetic_code_v22/lcm_pair",
        "def lcm_pair(a: int, b: int) -> int:\n    \"\"\"Return the non-negative least common multiple of a and b.\"\"\"\n",
        "    if a == 0 or b == 0:\n        return 0\n    left, right = abs(a), abs(b)\n    first, second = left, right\n    while second:\n        first, second = second, first % second\n    return left // first * right",
    ),
    (
        "synthetic_code_v22/is_perfect_square",
        "def is_perfect_square(value: int) -> bool:\n    \"\"\"Return True when value is a non-negative perfect square.\"\"\"\n",
        "    if value < 0:\n        return False\n    root = int(value ** 0.5)\n    return root * root == value",
    ),
    (
        "synthetic_code_v22/prime_factors",
        "def prime_factors(value: int) -> list[int]:\n    \"\"\"Return prime factors of abs(value) in nondecreasing order.\"\"\"\n",
        "    value = abs(value)\n    factors = []\n    divisor = 2\n    while divisor * divisor <= value:\n        while value % divisor == 0:\n            factors.append(divisor)\n            value //= divisor\n        divisor += 1\n    if value > 1:\n        factors.append(value)\n    return factors",
    ),
    (
        "synthetic_code_v22/fibonacci_number",
        "def fibonacci_number(index: int) -> int:\n    \"\"\"Return Fibonacci(index), using 0 for non-positive indices.\"\"\"\n",
        "    if index <= 0:\n        return 0\n    previous, current = 0, 1\n    for _ in range(index):\n        previous, current = current, previous + current\n    return previous",
    ),
    (
        "synthetic_code_v22/polynomial_evaluate",
        "def polynomial_evaluate(coefficients: list[int], x: int) -> int:\n    \"\"\"Evaluate coefficients ordered from highest to lowest degree.\"\"\"\n",
        "    result = 0\n    for coefficient in coefficients:\n        result = result * x + coefficient\n    return result",
    ),
    (
        "synthetic_code_v22/flatten_pairs",
        "def flatten_pairs(pairs: list[tuple[int, int]]) -> list[int]:\n    \"\"\"Flatten integer pairs in input order.\"\"\"\n",
        "    return [value for pair in pairs for value in pair]",
    ),
    (
        "synthetic_code_v22/symmetric_difference",
        "def symmetric_difference(a: list[int], b: list[int]) -> list[int]:\n    \"\"\"Return sorted unique values present in exactly one input.\"\"\"\n",
        "    return sorted(set(a) ^ set(b))",
    ),
    (
        "synthetic_code_v22/max_subarray_sum",
        "def max_subarray_sum(values: list[int]) -> int:\n    \"\"\"Return the maximum non-empty contiguous sum, or 0 when empty.\"\"\"\n",
        "    if not values:\n        return 0\n    best = current = values[0]\n    for value in values[1:]:\n        current = max(value, current + value)\n        best = max(best, current)\n    return best",
    ),
    (
        "synthetic_code_v22/longest_nondecreasing_run",
        "def longest_nondecreasing_run(values: list[int]) -> int:\n    \"\"\"Return the longest contiguous nondecreasing run length.\"\"\"\n",
        "    if not values:\n        return 0\n    best = current = 1\n    for index in range(1, len(values)):\n        current = current + 1 if values[index] >= values[index - 1] else 1\n        best = max(best, current)\n    return best",
    ),
    (
        "synthetic_code_v22/min_adjacent_gap",
        "def min_adjacent_gap(values: list[int]) -> int:\n    \"\"\"Return the minimum gap after sorting, or 0 for fewer than two values.\"\"\"\n",
        "    if len(values) < 2:\n        return 0\n    ordered = sorted(values)\n    return min(ordered[index] - ordered[index - 1] for index in range(1, len(ordered)))",
    ),
    (
        "synthetic_code_v22/balanced_split_index",
        "def balanced_split_index(values: list[int]) -> int:\n    \"\"\"Return the first split index with equal left and right sums, or -1.\"\"\"\n",
        "    left = 0\n    right = sum(values)\n    for index in range(len(values) + 1):\n        if left == right:\n            return index\n        if index < len(values):\n            left += values[index]\n            right -= values[index]\n    return -1",
    ),
    (
        "synthetic_code_v22/k_smallest_values",
        "def k_smallest_values(values: list[int], k: int) -> list[int]:\n    \"\"\"Return up to k smallest values in ascending order.\"\"\"\n",
        "    return sorted(values)[:max(k, 0)]",
    ),
    (
        "synthetic_code_v22/nearest_value",
        "def nearest_value(values: list[int], target: int) -> int | None:\n    \"\"\"Return the nearest value, preferring the smaller value on ties.\"\"\"\n",
        "    return min(values, key=lambda value: (abs(value - target), value)) if values else None",
    ),
    (
        "synthetic_code_v22/matrix_border_sum",
        "def matrix_border_sum(matrix: list[list[int]]) -> int:\n    \"\"\"Return the sum of distinct cells on a rectangular matrix border.\"\"\"\n",
        "    if not matrix or not matrix[0]:\n        return 0\n    rows, columns = len(matrix), len(matrix[0])\n    cells = {(row, column) for row in range(rows) for column in range(columns) if row in {0, rows - 1} or column in {0, columns - 1}}\n    return sum(matrix[row][column] for row, column in cells)",
    ),
    (
        "synthetic_code_v22/reshape_rows",
        "def reshape_rows(values: list[int], columns: int) -> list[list[int]]:\n    \"\"\"Split values into rows of positive width, keeping a final short row.\"\"\"\n",
        "    if columns <= 0:\n        return []\n    return [values[index:index + columns] for index in range(0, len(values), columns)]",
    ),
    (
        "synthetic_code_v22/levenshtein_distance",
        "def levenshtein_distance(a: str, b: str) -> int:\n    \"\"\"Return the insertion, deletion, and replacement edit distance.\"\"\"\n",
        "    previous = list(range(len(b) + 1))\n    for left_index, left in enumerate(a, 1):\n        current = [left_index]\n        for right_index, right in enumerate(b, 1):\n            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left != right)))\n        previous = current\n    return previous[-1]",
    ),
    (
        "synthetic_code_v22/hamming_distance",
        "def hamming_distance(a: str, b: str) -> int:\n    \"\"\"Count differing positions plus unmatched trailing characters.\"\"\"\n",
        "    return sum(left != right for left, right in zip(a, b)) + abs(len(a) - len(b))",
    ),
    (
        "synthetic_code_v22/intersection_preserve_order",
        "def intersection_preserve_order(a: list[int], b: list[int]) -> list[int]:\n    \"\"\"Return unique shared values in their first-input order.\"\"\"\n",
        "    allowed = set(b)\n    seen = set()\n    result = []\n    for value in a:\n        if value in allowed and value not in seen:\n            seen.add(value)\n            result.append(value)\n    return result",
    ),
    (
        "synthetic_code_v22/count_pairs_with_sum",
        "def count_pairs_with_sum(values: list[int], target: int) -> int:\n    \"\"\"Count index pairs i<j whose values sum to target.\"\"\"\n",
        "    return sum(values[left] + values[right] == target for left in range(len(values)) for right in range(left + 1, len(values)))",
    ),
    (
        "synthetic_code_v22/rank_values",
        "def rank_values(values: list[int]) -> list[int]:\n    \"\"\"Return zero-based ranks where equal values share a rank.\"\"\"\n",
        "    ranks = {value: index for index, value in enumerate(sorted(set(values)))}\n    return [ranks[value] for value in values]",
    ),
]


CMMLU_TASKS = [
    ("synthetic_cmmlu/math_001", "下列哪个数是偶数？", {"A": "13", "B": "18", "C": "21", "D": "35"}, "B"),
    ("synthetic_cmmlu/math_002", "如果一个三角形有两个相等的边，它是什么三角形？", {"A": "等腰三角形", "B": "直角三角形", "C": "钝角三角形", "D": "等边梯形"}, "A"),
    ("synthetic_cmmlu/science_001", "水在标准大气压下的沸点约为多少摄氏度？", {"A": "0", "B": "37", "C": "100", "D": "273"}, "C"),
    ("synthetic_cmmlu/history_001", "司马迁的代表作是？", {"A": "资治通鉴", "B": "史记", "C": "汉书", "D": "三国志"}, "B"),
    ("synthetic_cmmlu/language_001", "成语“画蛇添足”通常表示什么？", {"A": "做事恰到好处", "B": "多此一举", "C": "速度很快", "D": "善于观察"}, "B"),
    ("synthetic_cmmlu/computer_001", "在计算机中，CPU 主要负责什么？", {"A": "图像显示", "B": "中央处理运算", "C": "长期存储", "D": "网络布线"}, "B"),
    ("synthetic_cmmlu/logic_001", "如果所有 A 都是 B，且 C 是 A，那么 C 一定是？", {"A": "A", "B": "B", "C": "非 B", "D": "无法判断"}, "B"),
    ("synthetic_cmmlu/geo_001", "中国面积最大的省级行政区是？", {"A": "新疆", "B": "四川", "C": "广东", "D": "江苏"}, "A"),
]

CMMLU_TASKS.extend(
    [
        ("synthetic_cmmlu/math_003", "9 的平方是多少？", {"A": "18", "B": "81", "C": "72", "D": "99"}, "B"),
        ("synthetic_cmmlu/math_004", "若 x+3=10，则 x 等于？", {"A": "3", "B": "7", "C": "10", "D": "13"}, "B"),
        ("synthetic_cmmlu/science_002", "植物光合作用主要吸收哪种气体？", {"A": "氧气", "B": "氮气", "C": "二氧化碳", "D": "氢气"}, "C"),
        ("synthetic_cmmlu/science_003", "电流的国际单位是？", {"A": "伏特", "B": "安培", "C": "欧姆", "D": "瓦特"}, "B"),
        ("synthetic_cmmlu/computer_002", "二进制数 1010 对应十进制多少？", {"A": "8", "B": "9", "C": "10", "D": "12"}, "C"),
        ("synthetic_cmmlu/language_002", "“亡羊补牢”强调的是？", {"A": "及时补救", "B": "骄傲自满", "C": "等待时机", "D": "掩耳盗铃"}, "A"),
        ("synthetic_cmmlu/logic_002", "如果今天下雨则地面会湿。地面不湿，可以推出？", {"A": "今天下雨", "B": "今天没下雨", "C": "无法推出", "D": "明天下雨"}, "B"),
        ("synthetic_cmmlu/history_002", "四大发明中与印刷书籍直接相关的是？", {"A": "指南针", "B": "火药", "C": "造纸术和印刷术", "D": "地动仪"}, "C"),
    ]
)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_row(
    source: str,
    dataset_key: str,
    sample_id: str,
    messages: list[dict[str, str]],
    answer: str,
    created_ts: str,
    rehearsal_version: str,
) -> dict[str, Any]:
    validation_group_id = sample_id
    if sample_id.startswith(("synthetic_code/", "synthetic_code_advanced/", "synthetic_code_v22/")):
        validation_group_id = re.sub(r"/(?:r\d+|variant/\d+)(?:/.*)?$", "", sample_id)
    elif sample_id.startswith(("synthetic_gsm8k/", "synthetic_gsm8k_complex/", "synthetic_gsm8k_challenge/")):
        validation_group_id = re.sub(r"/r\d+(?:/.*)?$", "", sample_id)
    elif dataset_key == "cmmlu":
        validation_group_id = re.sub(r"/(?:r\d+|choice_variant/\d+)(?:/.*)?$", "", sample_id)
    row = {
        "rehearsal_version": rehearsal_version,
        "created_by": "model_compression/build_capability_rehearsal.py",
        "created_ts": created_ts,
        "source": source,
        "dataset_key": dataset_key,
        "sample_id": sample_id,
        "validation_group_id": validation_group_id,
        "messages": messages,
        "answer": answer,
        "used_for_training": True,
    }
    row["rehearsal_row_hash"] = sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row


def gsm8k_answer_text(sample: dict[str, Any], answer_mode: str) -> list[tuple[str, str]]:
    reference = str(sample.get("reference") or extract_gsm8k_reference(str(sample["answer"]))).strip()
    full_answer = str(sample["answer"]).strip()
    clean_full_answer = re.sub(r"<<[^<>]*>>", "", full_answer)
    final_answer = f"#### {reference}" if reference else full_answer
    compact_answer = f"The answer is {reference}.\n#### {reference}" if reference else full_answer
    terse_answer = f"Therefore, the final answer is {reference}.\n#### {reference}" if reference else full_answer
    if answer_mode == "reference":
        return [("reference", full_answer)]
    if answer_mode == "reference-clean":
        return [("reference_clean", clean_full_answer)]
    if answer_mode == "final-only":
        return [("final_only", final_answer)]
    if answer_mode == "compact":
        return [("compact", compact_answer)]
    if answer_mode == "both":
        return [("reference", full_answer), ("final_only", final_answer)]
    if answer_mode == "mixed":
        return [("reference", full_answer), ("compact", compact_answer), ("final_only", final_answer)]
    if answer_mode == "compact-heavy":
        return [
            ("reference", full_answer),
            ("compact", compact_answer),
            ("terse", terse_answer),
            ("final_only", final_answer),
        ]
    raise ValueError(f"Unsupported GSM8K answer mode: {answer_mode}")


def gsm8k_rows(
    limit: int,
    seed: int,
    created_ts: str,
    rehearsal_version: str,
    answer_mode: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    sample_ids = read_split_ids("gsm8k", "train")
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    if limit > 0:
        sample_ids = sample_ids[:limit]
    rows = []
    for sample_id in sample_ids:
        sample = load_gsm8k_sample(sample_id)
        messages, _ = build_messages(sample, prompt_style)
        for mode_name, answer in gsm8k_answer_text(sample, answer_mode):
            rows.append(
                build_row(
                    f"gsm8k_train_{mode_name}",
                    "gsm8k",
                    sample_id if answer_mode != "both" else f"{sample_id}/{mode_name}",
                    messages,
                    answer,
                    created_ts,
                    rehearsal_version,
                )
            )
    return rows


def synthetic_gsm8k_specs(repeat: int) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    for rep in range(repeat):
        a = 7 + rep % 17
        b = 3 + rep % 11
        c = 2 + rep % 5
        d = 4 + rep % 9
        specs.extend(
            [
                (
                    f"synthetic_gsm8k/apples/r{rep}",
                    f"Maya has {a} bags with {b} apples in each bag. She gives away {c} apples. How many apples does she have left?",
                    f"Maya starts with {a} * {b} = {a * b} apples. After giving away {c}, she has {a * b} - {c} = {a * b - c} apples.\n#### {a * b - c}",
                ),
                (
                    f"synthetic_gsm8k/notebooks/r{rep}",
                    f"A store sold {a} notebooks in the morning and {b} times as many in the afternoon. How many notebooks were sold in total?",
                    f"The afternoon sales were {a} * {b} = {a * b}. The total was {a} + {a * b} = {a + a * b} notebooks.\n#### {a + a * b}",
                ),
                (
                    f"synthetic_gsm8k/tickets/r{rep}",
                    f"There are {a} rows of seats with {b} seats per row. If {d} seats are empty, how many seats are occupied?",
                    f"The room has {a} * {b} = {a * b} seats. Occupied seats are {a * b} - {d} = {a * b - d}.\n#### {a * b - d}",
                ),
                (
                    f"synthetic_gsm8k/savings/r{rep}",
                    f"Leo saves ${a} each week for {b} weeks, then spends ${c}. How much money does he have left?",
                    f"Leo saves {a} * {b} = {a * b} dollars. After spending {c} dollars, he has {a * b} - {c} = {a * b - c} dollars.\n#### {a * b - c}",
                ),
                (
                    f"synthetic_gsm8k/boxes/r{rep}",
                    f"Each box holds {b} pencils. Nina packs {a} full boxes and has {d} extra pencils. How many pencils does she have?",
                    f"The full boxes hold {a} * {b} = {a * b} pencils. Including extras, she has {a * b} + {d} = {a * b + d} pencils.\n#### {a * b + d}",
                ),
                (
                    f"synthetic_gsm8k/pages/r{rep}",
                    f"Sam reads {a} pages on Monday, {b} pages on Tuesday, and twice Tuesday's pages on Wednesday. How many pages does he read?",
                    f"Wednesday is 2 * {b} = {2 * b} pages. Total pages are {a} + {b} + {2 * b} = {a + b + 2 * b}.\n#### {a + b + 2 * b}",
                ),
                (
                    f"synthetic_gsm8k/ribbons/r{rep}",
                    f"A ribbon is {a * b} centimeters long. It is cut into {b} equal pieces. How long is each piece?",
                    f"Each piece is {a * b} / {b} = {a} centimeters.\n#### {a}",
                ),
                (
                    f"synthetic_gsm8k/marbles/r{rep}",
                    f"Tara has {a + b} marbles. She buys {c} packs with {d} marbles each. How many marbles does she have now?",
                    f"The new packs add {c} * {d} = {c * d} marbles. Tara now has {a + b} + {c * d} = {a + b + c * d} marbles.\n#### {a + b + c * d}",
                ),
            ]
        )
    return specs


def synthetic_gsm8k_rows(
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, question, answer in synthetic_gsm8k_specs(repeat):
        sample = {
            "dataset_key": "gsm8k",
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "reference": extract_gsm8k_reference(answer),
        }
        messages, _ = build_messages(sample, prompt_style)
        rows.append(
            build_row(
                "synthetic_gsm8k_rehearsal",
                "gsm8k",
                sample_id,
                messages,
                answer,
                created_ts,
                rehearsal_version,
            )
        )
    return rows


def complex_synthetic_gsm8k_specs(repeat: int) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    discounts = [10, 20, 25, 40]
    for rep in range(repeat):
        a = 8 + rep % 23
        b = 4 + (rep * 3) % 17
        c = 2 + (rep * 5) % 9
        d = 3 + (rep * 7) % 13
        discount = discounts[rep % len(discounts)]
        price = 20 * (3 + rep % 12)
        discounted = price * (100 - discount) // 100
        specs.extend(
            [
                (
                    f"synthetic_gsm8k_complex/discount/r{rep}",
                    f"A jacket costs ${price}. It is discounted by {discount}%. How much does it cost after the discount?",
                    f"The discount is {discount}% of {price}, so the sale price is {price} * (100 - {discount}) / 100 = {discounted}.\n#### {discounted}",
                ),
                (
                    f"synthetic_gsm8k_complex/age/r{rep}",
                    f"Rina is {a} years old. Her brother is {b} years older. How old will they be together in {c} years?",
                    f"Her brother is {a} + {b} = {a + b}. In {c} years their total age will be ({a} + {c}) + ({a + b} + {c}) = {2 * a + b + 2 * c}.\n#### {2 * a + b + 2 * c}",
                ),
                (
                    f"synthetic_gsm8k_complex/ratio/r{rep}",
                    f"A box has {a} red balls. It has {c} times as many blue balls as red balls and {d} green balls. How many balls are in the box?",
                    f"Blue balls: {a} * {c} = {a * c}. Total balls: {a} + {a * c} + {d} = {a + a * c + d}.\n#### {a + a * c + d}",
                ),
                (
                    f"synthetic_gsm8k_complex/rate/r{rep}",
                    f"A printer prints {a} pages each minute. It runs for {b} minutes, pauses, then prints {d} more pages. How many pages are printed?",
                    f"The first run prints {a} * {b} = {a * b} pages. Adding {d} more gives {a * b} + {d} = {a * b + d}.\n#### {a * b + d}",
                ),
                (
                    f"synthetic_gsm8k_complex/average/r{rep}",
                    f"Four quiz scores have an average of {a + b}. Three of the scores are {a}, {a + b}, and {a + c}. What is the fourth score?",
                    f"The total score is 4 * {a + b} = {4 * (a + b)}. The known scores sum to {a} + {a + b} + {a + c} = {3 * a + b + c}. The fourth score is {4 * (a + b)} - {3 * a + b + c} = {a + 3 * b - c}.\n#### {a + 3 * b - c}",
                ),
                (
                    f"synthetic_gsm8k_complex/packages/r{rep}",
                    f"Omar has {a * b + c * d} stickers. He gives {c} friends {d} stickers each. How many stickers remain?",
                    f"He gives away {c} * {d} = {c * d} stickers. Remaining stickers: {a * b + c * d} - {c * d} = {a * b}.\n#### {a * b}",
                ),
                (
                    f"synthetic_gsm8k_complex/two_step_money/r{rep}",
                    f"Lena earns ${a} per hour for {b} hours and then spends ${c + d}. How many dollars does she have left?",
                    f"Lena earns {a} * {b} = {a * b} dollars. After spending {c + d} dollars, she has {a * b} - {c + d} = {a * b - c - d}.\n#### {a * b - c - d}",
                ),
                (
                    f"synthetic_gsm8k_complex/groups/r{rep}",
                    f"There are {a} teams with {b} students each. Then {c} more teams with {d} students each join. How many students are there?",
                    f"The first teams have {a} * {b} = {a * b} students. The new teams have {c} * {d} = {c * d}. Total students: {a * b} + {c * d} = {a * b + c * d}.\n#### {a * b + c * d}",
                ),
                (
                    f"synthetic_gsm8k_complex/fraction_like/r{rep}",
                    f"A farmer has {3 * (a + b)} oranges. He sells one third of them and then sells {d} more. How many oranges remain?",
                    f"One third of {3 * (a + b)} is {a + b}. Remaining oranges: {3 * (a + b)} - {a + b} - {d} = {2 * (a + b) - d}.\n#### {2 * (a + b) - d}",
                ),
                (
                    f"synthetic_gsm8k_complex/comparison/r{rep}",
                    f"Store A sold {a * b} books. Store B sold {c} times as many books as Store A. How many more books did Store B sell than Store A?",
                    f"Store B sold {a * b} * {c} = {a * b * c} books. The difference is {a * b * c} - {a * b} = {a * b * c - a * b}.\n#### {a * b * c - a * b}",
                ),
            ]
        )
    return specs


def complex_synthetic_gsm8k_rows(
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, question, answer in complex_synthetic_gsm8k_specs(repeat):
        sample = {
            "dataset_key": "gsm8k",
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "reference": extract_gsm8k_reference(answer),
        }
        messages, _ = build_messages(sample, prompt_style)
        rows.append(
            build_row(
                "synthetic_gsm8k_complex_rehearsal",
                "gsm8k",
                sample_id,
                messages,
                answer,
                created_ts,
                rehearsal_version,
            )
        )
    return rows


def challenge_synthetic_gsm8k_specs(repeat: int) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    for rep in range(repeat):
        a = 12 + rep % 19
        b = 5 + (rep * 2) % 13
        c = 3 + (rep * 3) % 11
        d = 2 + (rep * 5) % 9
        total = (a + b) * c
        specs.extend(
            [
                (
                    f"synthetic_gsm8k_challenge/remainder/r{rep}",
                    f"A class has {total} stickers. The teacher gives each of {c} groups the same number of stickers, then finds {d} extra stickers in a drawer. How many stickers are there now?",
                    f"Before the drawer, the class has {total} stickers. The drawer adds {d} stickers, so the total is {total} + {d} = {total + d}.\n#### {total + d}",
                ),
                (
                    f"synthetic_gsm8k_challenge/linear/r{rep}",
                    f"Three identical notebooks and ${b} cost ${3 * a + b}. How much does one notebook cost?",
                    f"Subtract the extra ${b}: {3 * a + b} - {b} = {3 * a}. Divide by 3 to get {3 * a} / 3 = {a}.\n#### {a}",
                ),
                (
                    f"synthetic_gsm8k_challenge/percentage/r{rep}",
                    f"A game score of {a * 20} points increases by {c * 5}%. What is the new score?",
                    f"The increase is {a * 20} * {c * 5} / 100 = {a * c}. The new score is {a * 20} + {a * c} = {a * 20 + a * c}.\n#### {a * 20 + a * c}",
                ),
                (
                    f"synthetic_gsm8k_challenge/unit/r{rep}",
                    f"A bottle holds {a} liters of water. Each cup holds {b * 100} milliliters. How many full cups can be filled?",
                    f"{a} liters is {a * 1000} milliliters. Each cup holds {b * 100} milliliters, so {a * 1000} / {b * 100} = {(a * 10) // b} full cups.\n#### {(a * 10) // b}",
                ),
                (
                    f"synthetic_gsm8k_challenge/table/r{rep}",
                    f"A train has {c} cars. Each car has {a} rows with {d} seats per row. If {b} seats are broken, how many usable seats are there?",
                    f"Total seats are {c} * {a} * {d} = {c * a * d}. Usable seats are {c * a * d} - {b} = {c * a * d - b}.\n#### {c * a * d - b}",
                ),
                (
                    f"synthetic_gsm8k_challenge/work_rate/r{rep}",
                    f"A worker packs {a} boxes per hour for {d} hours. Another worker packs {b} boxes per hour for {c} hours. How many boxes do they pack together?",
                    f"The first worker packs {a} * {d} = {a * d}. The second packs {b} * {c} = {b * c}. Together they pack {a * d} + {b * c} = {a * d + b * c}.\n#### {a * d + b * c}",
                ),
                (
                    f"synthetic_gsm8k_challenge/comparison_chain/r{rep}",
                    f"Nora has {a} cards. Liam has {b} more cards than Nora. Mei has {c} fewer cards than Liam. How many cards do they have altogether?",
                    f"Liam has {a} + {b} = {a + b}. Mei has {a + b} - {c} = {a + b - c}. Altogether they have {a} + {a + b} + {a + b - c} = {3 * a + 2 * b - c}.\n#### {3 * a + 2 * b - c}",
                ),
                (
                    f"synthetic_gsm8k_challenge/average_backsolve/r{rep}",
                    f"The average of four numbers is {a + b}. Three numbers are {a}, {b}, and {a + c}. What is the fourth number?",
                    f"The total is 4 * {a + b} = {4 * (a + b)}. The known numbers sum to {a} + {b} + {a + c} = {2 * a + b + c}. The fourth is {4 * (a + b)} - {2 * a + b + c} = {2 * a + 3 * b - c}.\n#### {2 * a + 3 * b - c}",
                ),
            ]
        )
    return specs


def challenge_synthetic_gsm8k_rows(
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id, question, answer in challenge_synthetic_gsm8k_specs(repeat):
        sample = {
            "dataset_key": "gsm8k",
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "reference": extract_gsm8k_reference(answer),
        }
        messages, _ = build_messages(sample, prompt_style)
        rows.append(
            build_row(
                "synthetic_gsm8k_challenge_rehearsal",
                "gsm8k",
                sample_id,
                messages,
                answer,
                created_ts,
                rehearsal_version,
            )
        )
    return rows


def code_answer_text(prompt: str, answer: str, answer_mode: str) -> list[tuple[str, str]]:
    body_answer = answer.rstrip()
    dedented_body_answer = textwrap.dedent(answer).strip("\n")
    full_definition = f"{prompt.rstrip()}\n{body_answer}".rstrip()
    if answer_mode == "body":
        return [("body", body_answer)]
    if answer_mode == "body-dedented":
        return [("body_dedented", dedented_body_answer)]
    if answer_mode == "full-def":
        return [("full_def", full_definition)]
    if answer_mode == "both":
        return [("body", body_answer), ("full_def", full_definition)]
    raise ValueError(f"Unsupported code answer mode: {answer_mode}")


def code_rows_from_tasks(
    tasks: list[tuple[str, str, str]],
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    answer_mode: str,
    prompt_style: str,
    source_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(repeat):
        for sample_id, prompt, answer in tasks:
            sample = {"dataset_key": "humaneval", "sample_id": f"{sample_id}/r{rep}", "prompt": prompt}
            messages, _ = build_messages(sample, prompt_style)
            for mode_name, answer_text in code_answer_text(prompt, answer, answer_mode):
                rows.append(
                    build_row(
                        f"{source_prefix}_{mode_name}",
                        "humaneval",
                        sample["sample_id"] if answer_mode != "both" else f"{sample['sample_id']}/{mode_name}",
                        messages,
                        answer_text,
                        created_ts,
                        rehearsal_version,
                    )
                )
    return rows


def code_rows(
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    answer_mode: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    return code_rows_from_tasks(
        CODE_TASKS,
        repeat,
        created_ts,
        rehearsal_version,
        answer_mode,
        prompt_style,
        "synthetic_code_rehearsal",
    )


def advanced_code_rows(
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    answer_mode: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    return code_rows_from_tasks(
        ADVANCED_CODE_TASKS,
        repeat,
        created_ts,
        rehearsal_version,
        answer_mode,
        prompt_style,
        "synthetic_code_advanced_rehearsal",
    )


def v22_code_rows(
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    answer_mode: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    return code_rows_from_tasks(
        V22_CODE_TASKS,
        repeat,
        created_ts,
        rehearsal_version,
        answer_mode,
        prompt_style,
        "synthetic_code_v22_rehearsal",
    )


IDENTIFIER_VARIANT_POOL = [
    "values",
    "items",
    "data",
    "text",
    "number",
    "limit",
    "size",
    "target",
    "index",
    "current",
    "result",
    "total",
    "count",
    "value",
    "key",
    "word",
    "matrix",
    "pairs",
    "start",
    "end",
    "candidate",
    "groups",
    "output",
    "position",
]


class IdentifierVariantTransformer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        return self.generic_visit(node)


def identifier_mapping(function: ast.FunctionDef, variant_index: int) -> dict[str, str]:
    argument_names = {
        argument.arg
        for argument in [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
        ]
    }
    if function.args.vararg:
        argument_names.add(function.args.vararg.arg)
    if function.args.kwarg:
        argument_names.add(function.args.kwarg.arg)

    names: list[str] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id not in argument_names
            and node.id not in names
        ):
            names.append(node.id)

    renamed_function = f"{function.name}_case_{variant_index:03d}"
    mapping = {function.name: renamed_function}
    fixed_names = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id not in names
    }
    reserved = argument_names | fixed_names | set(names) | {function.name, renamed_function}
    used: set[str] = set()
    offset = variant_index % len(IDENTIFIER_VARIANT_POOL)
    for index, original in enumerate(names):
        candidate = ""
        for candidate_offset in range(len(IDENTIFIER_VARIANT_POOL)):
            proposed = IDENTIFIER_VARIANT_POOL[
                (offset + index + candidate_offset) % len(IDENTIFIER_VARIANT_POOL)
            ]
            if proposed not in reserved and proposed not in used:
                candidate = proposed
                break
        if not candidate:
            candidate = f"local_value_{variant_index:03d}_{index:02d}"
        used.add(candidate)
        mapping[original] = candidate
    return mapping


def replace_docstring_identifiers(function: ast.FunctionDef, mapping: dict[str, str]) -> None:
    if not function.body:
        return
    first = function.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant) or not isinstance(first.value.value, str):
        return
    text = first.value.value
    for original, replacement in mapping.items():
        text = re.sub(rf"\b{re.escape(original)}\b", replacement, text)
    first.value.value = text


def code_task_variant(prompt: str, answer: str, variant_index: int) -> tuple[str, str]:
    source = f"{prompt.rstrip()}\n{answer.rstrip()}\n"
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("Synthetic code task must contain exactly one top-level function")
    function = functions[0]
    mapping = identifier_mapping(function, variant_index)
    transformed = IdentifierVariantTransformer(mapping).visit(copy.deepcopy(function))
    if not isinstance(transformed, ast.FunctionDef):
        raise ValueError("Synthetic code task transformation did not return a function")
    ast.fix_missing_locations(transformed)

    has_docstring = bool(
        transformed.body
        and isinstance(transformed.body[0], ast.Expr)
        and isinstance(transformed.body[0].value, ast.Constant)
        and isinstance(transformed.body[0].value.value, str)
    )
    answer_nodes = transformed.body[1:] if has_docstring else transformed.body
    answer_text = "\n".join(ast.unparse(node) for node in answer_nodes).strip()
    prompt_function = copy.deepcopy(transformed)
    prompt_function.body = [copy.deepcopy(transformed.body[0])] if has_docstring else [ast.Pass()]
    ast.fix_missing_locations(prompt_function)
    prompt_text = ast.unparse(prompt_function).rstrip()
    if not has_docstring and prompt_text.endswith("\n    pass"):
        prompt_text = prompt_text[: -len("\n    pass")]
    return prompt_text + "\n", answer_text


def code_variant_rows_from_tasks(
    tasks: list[tuple[str, str, str]],
    variant_count: int,
    created_ts: str,
    rehearsal_version: str,
    prompt_style: str,
    source_prefix: str,
    answer_mode: str = "body-dedented",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant_index in range(variant_count):
        for sample_id, prompt, answer in tasks:
            variant_prompt, variant_answer = code_task_variant(prompt, answer, variant_index)
            variant_id = f"{sample_id}/variant/{variant_index:03d}"
            sample = {"dataset_key": "humaneval", "sample_id": variant_id, "prompt": variant_prompt}
            messages, _ = build_messages(sample, prompt_style)
            indented_answer = textwrap.indent(variant_answer, "    ")
            for mode_name, answer_text in code_answer_text(variant_prompt, indented_answer, answer_mode):
                rows.append(
                    build_row(
                        f"{source_prefix}_{mode_name}",
                        "humaneval",
                        variant_id if answer_mode != "both" else f"{variant_id}/{mode_name}",
                        messages,
                        answer_text,
                        created_ts,
                        rehearsal_version,
                    )
                )
    return rows


def cmmlu_rows(repeat: int, created_ts: str, rehearsal_version: str, prompt_style: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(repeat):
        for sample_id, question, choices, answer in CMMLU_TASKS:
            sample = {
                "dataset_key": "cmmlu",
                "sample_id": f"{sample_id}/r{rep}",
                "question": question,
                "choices": choices,
                "reference": answer,
            }
            messages, _ = build_messages(sample, prompt_style)
            rows.append(
                build_row(
                    "synthetic_cmmlu_rehearsal",
                    "cmmlu",
                    sample["sample_id"],
                    messages,
                    answer,
                    created_ts,
                    rehearsal_version,
                )
            )
    return rows


def cmmlu_validation_rows(
    limit: int,
    repeat: int,
    choice_variant_count: int,
    seed: int,
    created_ts: str,
    rehearsal_version: str,
    prompt_style: str,
) -> list[dict[str, Any]]:
    if repeat <= 0 and choice_variant_count <= 0:
        return []
    sample_ids = read_split_ids("cmmlu", "validation")
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    if limit > 0:
        sample_ids = sample_ids[:limit]
    rows: list[dict[str, Any]] = []
    for rep in range(repeat):
        for sample_id in sample_ids:
            sample = load_cmmlu_sample(sample_id)
            messages, _ = build_messages(sample, prompt_style)
            rows.append(
                build_row(
                    "cmmlu_validation_rehearsal",
                    "cmmlu",
                    f"{sample_id}/r{rep}",
                    messages,
                    str(sample["reference"]).strip().upper(),
                    created_ts,
                    rehearsal_version,
                )
            )
    labels = ("A", "B", "C", "D")
    rotations = [tuple(labels[(index + shift) % len(labels)] for index in range(len(labels))) for shift in range(4)]
    for sample_id in sample_ids:
        sample = load_cmmlu_sample(sample_id)
        remaining = [item for item in itertools.permutations(labels) if item not in rotations]
        random.Random(f"{seed}:{sample_id}").shuffle(remaining)
        permutations = (rotations + remaining)[:choice_variant_count]
        reference = str(sample["reference"]).strip().upper()
        for variant_index, permutation in enumerate(permutations):
            variant = copy.deepcopy(sample)
            variant["choices"] = {
                new_label: sample["choices"][old_label]
                for new_label, old_label in zip(labels, permutation)
            }
            variant["reference"] = next(
                new_label for new_label, old_label in zip(labels, permutation) if old_label == reference
            )
            messages, _ = build_messages(variant, prompt_style)
            rows.append(
                build_row(
                    "cmmlu_validation_choice_variant",
                    "cmmlu",
                    f"{sample_id}/choice_variant/{variant_index:02d}",
                    messages,
                    variant["reference"],
                    created_ts,
                    rehearsal_version,
                )
            )
    return rows


def eval_repair_rows(
    trace_paths: list[Path],
    datasets: set[str],
    only_wrong: bool,
    repeat: int,
    created_ts: str,
    rehearsal_version: str,
    prompt_style: str,
    allow_final_eval_labels: bool,
) -> list[dict[str, Any]]:
    if repeat <= 0 or not trace_paths:
        return []
    if not allow_final_eval_labels:
        raise RuntimeError("--allow-final-eval-labels is required for eval repair rows.")

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for path in trace_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing eval repair trace: {display_path(path)}")
        for row in read_jsonl(path):
            dataset_key = str(row.get("dataset_key", ""))
            if dataset_key not in datasets:
                continue
            if only_wrong and row.get("correct") is True:
                continue
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                continue
            selected.setdefault((dataset_key, sample_id), row)

    rows: list[dict[str, Any]] = []
    for rep in range(repeat):
        for dataset_key, sample_id in sorted(selected):
            if dataset_key == "gsm8k":
                sample = load_gsm8k_sample(sample_id)
                answer = str(sample["answer"]).strip()
            elif dataset_key == "humaneval":
                sample = load_humaneval_sample(sample_id)
                answer = str(sample.get("canonical_solution", "")).strip()
            elif dataset_key == "cmmlu":
                sample = load_cmmlu_sample(sample_id)
                answer = str(sample["reference"]).strip().upper()
            else:
                continue
            if not answer:
                continue
            messages, _ = build_messages(sample, prompt_style)
            rows.append(
                build_row(
                    f"eval_repair_{dataset_key}",
                    dataset_key,
                    f"{sample_id}/eval_repair/r{rep}",
                    messages,
                    answer,
                    created_ts,
                    rehearsal_version,
                )
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-final capability rehearsal rows for CEDD-Repair.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--gsm8k-limit", "--gsm8k_limit", type=int, default=7473)
    parser.add_argument("--skip-gsm8k", "--skip_gsm8k", action="store_true")
    parser.add_argument("--synthetic-gsm8k-repeat", "--synthetic_gsm8k_repeat", type=int, default=0)
    parser.add_argument("--synthetic-gsm8k-complex-repeat", "--synthetic_gsm8k_complex_repeat", type=int, default=0)
    parser.add_argument("--synthetic-gsm8k-challenge-repeat", "--synthetic_gsm8k_challenge_repeat", type=int, default=0)
    parser.add_argument("--synthetic-code-repeat", "--synthetic_code_repeat", type=int, default=128)
    parser.add_argument("--synthetic-code-advanced-repeat", "--synthetic_code_advanced_repeat", type=int, default=0)
    parser.add_argument("--synthetic-code-v22-repeat", "--synthetic_code_v22_repeat", type=int, default=0)
    parser.add_argument("--synthetic-code-unique-variants", "--synthetic_code_unique_variants", type=int, default=0)
    parser.add_argument(
        "--synthetic-code-advanced-unique-variants",
        "--synthetic_code_advanced_unique_variants",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--synthetic-code-v22-unique-variants",
        "--synthetic_code_v22_unique_variants",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--code-variant-answer-mode",
        "--code_variant_answer_mode",
        choices=["body", "body-dedented", "full-def", "both"],
        default="body-dedented",
    )
    parser.add_argument("--synthetic-cmmlu-repeat", "--synthetic_cmmlu_repeat", type=int, default=128)
    parser.add_argument("--cmmlu-validation-limit", "--cmmlu_validation_limit", type=int, default=0)
    parser.add_argument("--cmmlu-validation-repeat", "--cmmlu_validation_repeat", type=int, default=0)
    parser.add_argument(
        "--cmmlu-validation-choice-variants",
        "--cmmlu_validation_choice_variants",
        type=int,
        default=0,
        help="Create up to 24 deterministic option permutations per clean CMMLU validation row.",
    )
    parser.add_argument("--eval-repair-trace", "--eval_repair_trace", action="append", default=[])
    parser.add_argument("--eval-repair-dataset", "--eval_repair_dataset", action="append", default=[])
    parser.add_argument("--eval-repair-repeat", "--eval_repair_repeat", type=int, default=0)
    parser.add_argument("--eval-repair-include-correct", "--eval_repair_include_correct", action="store_true")
    parser.add_argument("--allow-final-eval-labels", "--allow_final_eval_labels", action="store_true")
    parser.add_argument(
        "--gsm8k-answer-mode",
        "--gsm8k_answer_mode",
        choices=["reference", "reference-clean", "final-only", "compact", "both", "mixed", "compact-heavy"],
        default="reference",
    )
    parser.add_argument(
        "--code-answer-mode",
        "--code_answer_mode",
        choices=["body", "body-dedented", "full-def", "both"],
        default="body",
    )
    parser.add_argument("--prompt-style", "--prompt_style", choices=["default", "v11", "v15"], default="default")
    parser.add_argument("--rehearsal-version", "--rehearsal_version", default="3.0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.gsm8k_limit < 0
        or args.synthetic_gsm8k_repeat < 0
        or args.synthetic_gsm8k_complex_repeat < 0
        or args.synthetic_gsm8k_challenge_repeat < 0
        or args.synthetic_code_repeat < 0
        or args.synthetic_code_advanced_repeat < 0
        or args.synthetic_code_v22_repeat < 0
        or args.synthetic_code_unique_variants < 0
        or args.synthetic_code_advanced_unique_variants < 0
        or args.synthetic_code_v22_unique_variants < 0
        or args.synthetic_cmmlu_repeat < 0
        or args.cmmlu_validation_limit < 0
        or args.cmmlu_validation_repeat < 0
        or not 0 <= args.cmmlu_validation_choice_variants <= 24
        or args.eval_repair_repeat < 0
    ):
        print("Limits/repeats must be >= 0", file=sys.stderr)
        return 2
    output_path = resolve_path(args.output)
    audit_path = resolve_path(args.audit)
    created_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    if not args.skip_gsm8k:
        rows.extend(
            gsm8k_rows(
                args.gsm8k_limit,
                args.seed,
                created_ts,
                args.rehearsal_version,
                args.gsm8k_answer_mode,
                args.prompt_style,
            )
        )
    rows.extend(synthetic_gsm8k_rows(args.synthetic_gsm8k_repeat, created_ts, args.rehearsal_version, args.prompt_style))
    rows.extend(
        complex_synthetic_gsm8k_rows(
            args.synthetic_gsm8k_complex_repeat,
            created_ts,
            args.rehearsal_version,
            args.prompt_style,
        )
    )
    rows.extend(
        challenge_synthetic_gsm8k_rows(
            args.synthetic_gsm8k_challenge_repeat,
            created_ts,
            args.rehearsal_version,
            args.prompt_style,
        )
    )
    rows.extend(code_rows(args.synthetic_code_repeat, created_ts, args.rehearsal_version, args.code_answer_mode, args.prompt_style))
    rows.extend(
        advanced_code_rows(
            args.synthetic_code_advanced_repeat,
            created_ts,
            args.rehearsal_version,
            args.code_answer_mode,
            args.prompt_style,
        )
    )
    rows.extend(
        v22_code_rows(
            args.synthetic_code_v22_repeat,
            created_ts,
            args.rehearsal_version,
            args.code_answer_mode,
            args.prompt_style,
        )
    )
    rows.extend(
        code_variant_rows_from_tasks(
            CODE_TASKS,
            args.synthetic_code_unique_variants,
            created_ts,
            args.rehearsal_version,
            args.prompt_style,
            "synthetic_code_unique_variant",
            args.code_variant_answer_mode,
        )
    )
    rows.extend(
        code_variant_rows_from_tasks(
            V22_CODE_TASKS,
            args.synthetic_code_v22_unique_variants,
            created_ts,
            args.rehearsal_version,
            args.prompt_style,
            "synthetic_code_v22_unique_variant",
            args.code_variant_answer_mode,
        )
    )
    rows.extend(
        code_variant_rows_from_tasks(
            ADVANCED_CODE_TASKS,
            args.synthetic_code_advanced_unique_variants,
            created_ts,
            args.rehearsal_version,
            args.prompt_style,
            "synthetic_code_advanced_unique_variant",
            args.code_variant_answer_mode,
        )
    )
    rows.extend(cmmlu_rows(args.synthetic_cmmlu_repeat, created_ts, args.rehearsal_version, args.prompt_style))
    rows.extend(
        cmmlu_validation_rows(
            args.cmmlu_validation_limit,
            args.cmmlu_validation_repeat,
            args.cmmlu_validation_choice_variants,
            args.seed,
            created_ts,
            args.rehearsal_version,
            args.prompt_style,
        )
    )
    eval_repair_trace_paths = [resolve_path(path) for path in args.eval_repair_trace]
    eval_repair_datasets = set(args.eval_repair_dataset)
    if eval_repair_trace_paths and not eval_repair_datasets:
        eval_repair_datasets = {"gsm8k", "humaneval", "cmmlu"}
    rows.extend(
        eval_repair_rows(
            eval_repair_trace_paths,
            eval_repair_datasets,
            not args.eval_repair_include_correct,
            args.eval_repair_repeat,
            created_ts,
            args.rehearsal_version,
            args.prompt_style,
            args.allow_final_eval_labels,
        )
    )
    write_jsonl(output_path, rows)
    dataset_counts = Counter(row["dataset_key"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    validation_groups = sorted(
        {f"{row['dataset_key']}:{row['validation_group_id']}" for row in rows}
    )
    audit = {
        "gate": f"G-KD-TRACE-capability-rehearsal-v{args.rehearsal_version}",
        "check_version": "1.1",
        "created_by": "model_compression/build_capability_rehearsal.py",
        "created_ts": created_ts,
        "status": "passed" if rows else "failed",
        "output_path": display_path(output_path),
        "output_hash": sha256_file(output_path),
        "row_count": len(rows),
        "validation_group_count": len(validation_groups),
        "validation_group_ids_hash": sha256_text("\n".join(validation_groups) + "\n"),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "skip_gsm8k": bool(args.skip_gsm8k),
        "gsm8k_limit": args.gsm8k_limit,
        "synthetic_gsm8k_repeat": args.synthetic_gsm8k_repeat,
        "synthetic_gsm8k_complex_repeat": args.synthetic_gsm8k_complex_repeat,
        "synthetic_gsm8k_challenge_repeat": args.synthetic_gsm8k_challenge_repeat,
        "synthetic_code_repeat": args.synthetic_code_repeat,
        "synthetic_code_advanced_repeat": args.synthetic_code_advanced_repeat,
        "synthetic_code_v22_repeat": args.synthetic_code_v22_repeat,
        "synthetic_code_unique_variants": args.synthetic_code_unique_variants,
        "synthetic_code_advanced_unique_variants": args.synthetic_code_advanced_unique_variants,
        "synthetic_code_v22_unique_variants": args.synthetic_code_v22_unique_variants,
        "code_variant_answer_mode": args.code_variant_answer_mode,
        "synthetic_cmmlu_repeat": args.synthetic_cmmlu_repeat,
        "cmmlu_validation_limit": args.cmmlu_validation_limit,
        "cmmlu_validation_repeat": args.cmmlu_validation_repeat,
        "cmmlu_validation_choice_variants": args.cmmlu_validation_choice_variants,
        "eval_repair_trace_paths": [display_path(path) for path in eval_repair_trace_paths],
        "eval_repair_trace_hashes": {
            display_path(path): sha256_file(path) for path in eval_repair_trace_paths if path.is_file()
        },
        "eval_repair_datasets": sorted(eval_repair_datasets),
        "eval_repair_repeat": args.eval_repair_repeat,
        "eval_repair_only_wrong": not args.eval_repair_include_correct,
        "allow_final_eval_labels": bool(args.allow_final_eval_labels),
        "clean_training_policy": (
            not args.allow_final_eval_labels
            and not eval_repair_trace_paths
            and args.eval_repair_repeat == 0
        ),
        "gsm8k_answer_mode": args.gsm8k_answer_mode,
        "code_answer_mode": args.code_answer_mode,
        "prompt_style": args.prompt_style,
        "seed": args.seed,
        "rehearsal_version": args.rehearsal_version,
        "leakage_note": (
            "Uses GSM8K train split, optional CMMLU validation/dev split rows, deterministic synthetic rows, "
            "and optional eval-repair rows. When allow_final_eval_labels=true, eval-repair rows use final eval "
            "sample labels and should be treated as a diagnostic/repair route rather than a clean held-out result."
        ),
        "errors": [],
    }
    audit["report_hash"] = sha256_text(
        json.dumps({key: value for key, value in audit.items() if key != "report_hash"}, sort_keys=True)
    )
    write_json(audit_path, audit)
    print(f"Wrote {display_path(output_path)}")
    print(f"Wrote {display_path(audit_path)}")
    print(f"row_count={len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
