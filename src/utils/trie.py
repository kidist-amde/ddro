# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional


class Trie:
    """
    A basic implementation of a Trie (prefix tree) for token sequences.
    Used to constrain decoding during sequence generation (e.g., in generative IR).
    """

    def __init__(self, sequences: Optional[List[List[int]]] = None):
        """
        Initialize the Trie and optionally populate it with sequences.

        Args:
            sequences (List[List[int]], optional): Initial list of token ID sequences.
        """
        self.trie_dict: Dict[int, dict] = {}
        self.len = 0

        if sequences:
            for sequence in sequences:
                self._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie: Optional[Trie] = None
        self.bos_token_id: Optional[int] = None

    def append(self, trie: 'Trie', bos_token_id: int):
        """
        Appends another trie for dynamic expansion after a BOS token.

        Args:
            trie (Trie): Another Trie to append.
            bos_token_id (int): Token ID that marks where the append should occur.
        """
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        """
        Add a new sequence to the trie.

        Args:
            sequence (List[int]): List of token IDs.
        """
        self._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]) -> List[int]:
        """
        Get valid next token IDs given a prefix sequence.

        Args:
            prefix_sequence (List[int]): The current sequence prefix.

        Returns:
            List[int]: Valid next tokens from this prefix.
        """
        return self._get_from_trie(prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id)

    @staticmethod
    def load_from_dict(trie_dict: Dict) -> 'Trie':
        """
        Create a Trie object from a dictionary (e.g., previously saved).

        Args:
            trie_dict (Dict): The nested dictionary representing a trie.

        Returns:
            Trie: Loaded Trie object.
        """
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        """
        Recursively adds a sequence into the trie.

        Args:
            sequence (List[int]): The token sequence to add.
            trie_dict (Dict): The current level of the trie.
        """
        if sequence:
            head, tail = sequence[0], sequence[1:]
            if head not in trie_dict:
                trie_dict[head] = {}
            Trie._add_to_trie(tail, trie_dict[head])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie: Optional['Trie'] = None,
        bos_token_id: Optional[int] = None
    ) -> List[int]:
        """
        Recursively retrieves valid next tokens given a prefix.

        Args:
            prefix_sequence (List[int]): Current token prefix.
            trie_dict (Dict): The trie dictionary.
            append_trie (Trie, optional): Another Trie to merge from BOS.
            bos_token_id (int, optional): BOS token ID to trigger merging.

        Returns:
            List[int]: List of valid next token IDs.
        """
        if not prefix_sequence:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:], trie_dict[prefix_sequence[0]], append_trie, bos_token_id
            )
        else:
            return append_trie.get(prefix_sequence) if append_trie else []

    def __iter__(self):
        """
        Yield all sequences stored in the trie.
        """
        def _traverse(prefix: List[int], node: Dict):
            if node:
                for token in node:
                    yield from _traverse(prefix + [token], node[token])
            else:
                yield prefix

        return _traverse([], self.trie_dict)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, prefix_sequence: List[int]) -> List[int]:
        return self.get(prefix_sequence)
