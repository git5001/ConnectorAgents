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
    sub_list_dicts = []
    sub_list_uuid_sets = []  # track just the UUIDs in each sub-list

    for s_list in sub_lists:
        d = defaultdict(lambda: {"indices": set(), "lengths": set()})
        uuid_set = set()
        for item in s_list:
            parts = item.split(":")
            if len(parts) != 3:
                continue  # skip malformed items
            uuid_val, idx_str, length_str = parts
            try:
                idx = int(idx_str)
                length = int(length_str)
            except ValueError:
                continue  # skip if counters not parseable

            d[uuid_val]["indices"].add(idx)
            d[uuid_val]["lengths"].add(length)
            uuid_set.add(uuid_val)

        sub_list_dicts.append(d)
        sub_list_uuid_sets.append(uuid_set)

    if not sub_list_uuid_sets:
        return []

    # 1) Find intersection of *all* UUIDs across sub-lists
    common_uuids = set.intersection(*sub_list_uuid_sets)

    # 2) For each UUID in the intersection, gather all counters from all sub-lists
    #    and check if we have a full 0..(L-1).
    final_uuids = []

    for uuid_val in common_uuids:
        all_indices = set()
        all_lengths = set()
        # Union the counters/lengths from each sub-list that has this UUID
        for d in sub_list_dicts:
            if uuid_val in d:
                all_indices |= d[uuid_val]["indices"]
                all_lengths |= d[uuid_val]["lengths"]

        if not all_lengths:
            continue  # no known length -> skip

        # If there's more than one length, we can choose to check them all
        # or assume they're consistent. Let's take the *max* length found,
        # and verify counters 0..(max_length-1).
        max_length = max(all_lengths)
        needed_indices = set(range(max_length))
        if needed_indices.issubset(all_indices):
            final_uuids.append(uuid_val)

    # Sort if you want a stable order
    final_uuids.sort()
    return final_uuids
