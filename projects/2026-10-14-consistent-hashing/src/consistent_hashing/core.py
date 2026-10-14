"""Consistent hashing with virtual nodes for distributed load balancing."""

from __future__ import annotations

import hashlib
from bisect import bisect_right


class ConsistentHash:
    """Hash ring for consistent distribution of keys to servers.

    Uses virtual nodes (replicas) to improve key distribution when servers
    join or leave the ring.
    """

    def __init__(self, replicas: int = 160) -> None:
        """Initialize the hash ring with a given number of virtual nodes per server.

        Args:
            replicas: Number of virtual nodes per server (default 160).
        """
        if replicas < 1:
            raise ValueError("replicas must be at least 1")
        self.replicas = replicas
        self.ring: dict[int, str] = {}
        self.sorted_keys: list[int] = []
        self.servers: set[str] = set()

    @staticmethod
    def _hash(key: str) -> int:
        """Hash a string to a 32-bit unsigned integer."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def add_server(self, server: str) -> None:
        """Add a server to the hash ring."""
        if server in self.servers:
            return
        self.servers.add(server)
        for i in range(self.replicas):
            virtual_key = f"{server}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = server
        self.sorted_keys = sorted(self.ring.keys())

    def remove_server(self, server: str) -> None:
        """Remove a server from the hash ring."""
        if server not in self.servers:
            return
        self.servers.discard(server)
        for i in range(self.replicas):
            virtual_key = f"{server}:{i}"
            hash_value = self._hash(virtual_key)
            del self.ring[hash_value]
        self.sorted_keys = sorted(self.ring.keys())

    def get_server(self, key: str) -> str | None:
        """Get the server responsible for a given key.

        Returns None if no servers are in the ring.
        """
        if not self.ring:
            return None
        hash_value = self._hash(key)
        index = bisect_right(self.sorted_keys, hash_value)
        if index == len(self.sorted_keys):
            index = 0
        return self.ring[self.sorted_keys[index]]

    def get_servers(self, key: str, count: int = 1) -> list[str]:
        """Get the top N servers for replication, starting from the one responsible for key.

        Returns distinct servers in ring order. Count is clamped to available servers.
        """
        if not self.ring:
            return []
        if count < 1:
            raise ValueError("count must be at least 1")
        hash_value = self._hash(key)
        index = bisect_right(self.sorted_keys, hash_value)
        if index == len(self.sorted_keys):
            index = 0
        result = []
        seen: set[str] = set()
        for _ in range(len(self.sorted_keys)):
            server = self.ring[self.sorted_keys[index]]
            if server not in seen:
                result.append(server)
                seen.add(server)
                if len(result) == count:
                    break
            index = (index + 1) % len(self.sorted_keys)
        return result
