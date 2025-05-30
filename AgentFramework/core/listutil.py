from collections import defaultdict
from typing import List, Dict, Set


def longest_common_sublist(data: Dict[str, List[str]]) -> List[str]:
    """
    Finds the longest common prefix among lists stored in a dictionary.

    Args:
        data (Dict[str, List[str]]): A dictionary where each key maps to a list of strings.

    Returns:
        List[str]: The longest common prefix found among all lists.
    """
    if not data:
        return []

    # Extract all lists from the dictionary
    lists = list(data.values())

    # Sort by length to ensure we check only within the shortest list
    lists.sort(key=len)

    # Get the shortest list to use as a reference
    shortest_list = lists[0]

    # Find the longest common prefix
    common_prefix = []
    for i in range(len(shortest_list)):
        prefix = shortest_list[:i + 1]  # Take the current prefix
        if all(lst[:i + 1] == prefix for lst in lists):  # Check in all lists
            common_prefix = prefix
        else:
            break  # Stop early if a mismatch is found

    return common_prefix


def compare_lists(list1: List[str], list2: List[str]) -> bool:
    """
    Compares elements of two lists up to the length of the shorter one.

    Args:
        list1 (List[str]): First list to compare.
        list2 (List[str]): Second list to compare.

    Returns:
        bool: True if all compared elements are equal, False otherwise.
    """
    min_length = min(len(list1), len(list2))  # Determine the shorter list length

    for i in range(min_length):
        if list1[i] != list2[i]:
            return False  # Return immediately on first mismatch

    return True  # Return True if all compared elements matched


def find_common_complete_uuids(sub_lists: List[List[str]]) -> List[str]:
    """
    Given multiple sub-lists, each containing items like "uuid:counter:length",
    return the list of UUIDs that are:
      1) Present in *every* sub-list (common).
      2) 'Complete' if we combine all pieces from all sub-lists.

    'Complete' means that for a UUID with length = L, across the union
    of all sub-lists, we have counters 0..(L-1).

    Example:
        sub_lists = [
            ["aaa:0:1", "bbb:0:2", "ccc:0:1"],
            ["aaa:0:1", "bbb:1:2", "ddd:0:1"]
        ]
        --> returns ["aaa", "bbb"]
    """

    # Parse each sub-list into a dict of:
    #   sub_list_dict[uuid] = {
    #       "indices": set([ ... counters ... ]),
    #       "lengths": set([ ... lengths ... ])
    #   }
    from collections import defaultdict
    from typing import List

    if not sub_lists:  # empty input guard
        return []

    # -----------------------------------------------
    # 1) Parse each sub-list -> per-UUID meta-data
    # -----------------------------------------------
    per_list_meta = []  # one dict per sub-list
    per_list_uuid_sets = []  # and the plain set of uuids

    for s_list in sub_lists:
        d = defaultdict(lambda: {"indices": set(), "lengths": set()})
        for item in s_list:
            try:
                uuid_val, idx_str, length_str = item.split(":")
                idx = int(idx_str)
                length = int(length_str)
            except (ValueError, IndexError):
                continue  # skip malformed entries

            d[uuid_val]["indices"].add(idx)
            d[uuid_val]["lengths"].add(length)
        per_list_meta.append(d)
        per_list_uuid_sets.append(set(d.keys()))

    # -------------------------------------------------
    # 2) Which UUIDs occur in *every* sub-list?
    # -------------------------------------------------
    common_uuids = set.intersection(*per_list_uuid_sets)

    # -------------------------------------------------
    # 3) Respect the sequence of the first sub-list
    # -------------------------------------------------
    ordered_common = []
    seen = set()
    for item in sub_lists[0]:
        uuid_val = item.split(":")[0]
        if uuid_val in common_uuids and uuid_val not in seen:
            ordered_common.append(uuid_val)
            seen.add(uuid_val)

    # -------------------------------------------------
    # 4) Apply the relaxed “same length” test
    # -------------------------------------------------
    result = []
    for uuid_val in ordered_common:
        all_lengths = set()
        for d in per_list_meta:
            all_lengths |= d.get(uuid_val, {}).get("lengths", set())

        # keep the UUID if every list uses the *same* length
        if len(all_lengths) == 1:
            result.append(uuid_val)

    return result


def find_last_non_one(items):
    """
    Find the last item in a parent list which has a len > 1
    and return it
    :param items: parents
    :return: the length
    """
    for item in reversed(items):
        parts = item.split(':')
        if len(parts) != 3:
            raise ValueError(f"Malformed item: {item}")
        try:
            val = int(parts[-1])
        except ValueError:
            raise ValueError(f"Invalid integer in item: {item}")
        if val != 1:
            return val
    return 1
