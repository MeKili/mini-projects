"""Tests for consistent hashing implementation."""

import pytest

from consistent_hashing import ConsistentHash


def test_init_valid_replicas() -> None:
    """Test initialization with valid replica counts."""
    hash_ring = ConsistentHash(replicas=160)
    assert hash_ring.replicas == 160
    assert len(hash_ring.servers) == 0
    assert len(hash_ring.ring) == 0


def test_init_invalid_replicas() -> None:
    """Test initialization with invalid replica count."""
    with pytest.raises(ValueError, match="replicas must be at least 1"):
        ConsistentHash(replicas=0)
    with pytest.raises(ValueError, match="replicas must be at least 1"):
        ConsistentHash(replicas=-1)


def test_single_server() -> None:
    """Test with a single server."""
    hash_ring = ConsistentHash(replicas=3)
    hash_ring.add_server("server1")
    assert "server1" in hash_ring.servers
    assert len(hash_ring.ring) == 3
    assert hash_ring.get_server("key1") == "server1"
    assert hash_ring.get_server("key2") == "server1"


def test_multiple_servers() -> None:
    """Test with multiple servers."""
    hash_ring = ConsistentHash(replicas=4)
    servers = ["server1", "server2", "server3"]
    for server in servers:
        hash_ring.add_server(server)
    assert len(hash_ring.servers) == 3
    assert len(hash_ring.ring) == 12  # 3 servers * 4 replicas
    assert hash_ring.get_server("key1") in servers
    assert hash_ring.get_server("key2") in servers


def test_add_duplicate_server() -> None:
    """Test adding the same server twice."""
    hash_ring = ConsistentHash(replicas=2)
    hash_ring.add_server("server1")
    initial_ring_size = len(hash_ring.ring)
    hash_ring.add_server("server1")
    assert len(hash_ring.ring) == initial_ring_size
    assert hash_ring.get_server("key1") == "server1"


def test_remove_server() -> None:
    """Test removing a server."""
    hash_ring = ConsistentHash(replicas=3)
    hash_ring.add_server("server1")
    hash_ring.add_server("server2")
    assert len(hash_ring.servers) == 2
    hash_ring.remove_server("server1")
    assert "server1" not in hash_ring.servers
    assert len(hash_ring.ring) == 3  # Only server2's replicas remain
    assert hash_ring.get_server("key1") == "server2"


def test_remove_nonexistent_server() -> None:
    """Test removing a server that doesn't exist."""
    hash_ring = ConsistentHash(replicas=2)
    hash_ring.add_server("server1")
    hash_ring.remove_server("server2")  # Should not raise
    assert len(hash_ring.servers) == 1


def test_empty_ring() -> None:
    """Test behavior with an empty ring."""
    hash_ring = ConsistentHash()
    assert hash_ring.get_server("key1") is None
    assert hash_ring.get_servers("key1") == []


def test_key_distribution() -> None:
    """Test that keys are distributed across multiple servers."""
    hash_ring = ConsistentHash(replicas=10)
    servers = ["server1", "server2", "server3"]
    for server in servers:
        hash_ring.add_server(server)
    distribution = {server: 0 for server in servers}
    for i in range(1000):
        key = f"key_{i}"
        server = hash_ring.get_server(key)
        assert server is not None
        distribution[server] += 1
    assert all(count > 0 for count in distribution.values())
    assert all(200 < count < 400 for count in distribution.values())


def test_key_stability() -> None:
    """Test that the same key always maps to the same server."""
    hash_ring = ConsistentHash(replicas=5)
    hash_ring.add_server("server1")
    hash_ring.add_server("server2")
    server1 = hash_ring.get_server("stable_key")
    for _ in range(100):
        assert hash_ring.get_server("stable_key") == server1


def test_server_removal_stability() -> None:
    """Test that most keys remain on their original server after removal."""
    hash_ring = ConsistentHash(replicas=20)
    servers = ["server1", "server2", "server3"]
    for server in servers:
        hash_ring.add_server(server)
    original_mapping = {f"key_{i}": hash_ring.get_server(f"key_{i}") for i in range(100)}
    hash_ring.remove_server("server3")
    stable_count = 0
    for key, original_server in original_mapping.items():
        new_server = hash_ring.get_server(key)
        if original_server == "server3":
            assert new_server in ("server1", "server2")
        else:
            if new_server == original_server:
                stable_count += 1
    assert stable_count > 50


def test_get_servers_single() -> None:
    """Test getting a single server for a key."""
    hash_ring = ConsistentHash(replicas=5)
    hash_ring.add_server("server1")
    result = hash_ring.get_servers("key1", count=1)
    assert result == ["server1"]


def test_get_servers_multiple() -> None:
    """Test getting multiple servers for replication."""
    hash_ring = ConsistentHash(replicas=10)
    servers = ["server1", "server2", "server3"]
    for server in servers:
        hash_ring.add_server(server)
    result = hash_ring.get_servers("key1", count=2)
    assert len(result) == 2
    assert len(set(result)) == 2  # All distinct
    assert all(s in servers for s in result)


def test_get_servers_exceeds_available() -> None:
    """Test requesting more servers than available."""
    hash_ring = ConsistentHash(replicas=5)
    hash_ring.add_server("server1")
    hash_ring.add_server("server2")
    result = hash_ring.get_servers("key1", count=10)
    assert len(result) == 2
    assert set(result) == {"server1", "server2"}


def test_get_servers_invalid_count() -> None:
    """Test invalid count for get_servers."""
    hash_ring = ConsistentHash(replicas=5)
    hash_ring.add_server("server1")
    with pytest.raises(ValueError, match="count must be at least 1"):
        hash_ring.get_servers("key1", count=0)
    with pytest.raises(ValueError, match="count must be at least 1"):
        hash_ring.get_servers("key1", count=-1)


def test_hash_consistency() -> None:
    """Test that the hash function is consistent."""
    hash_ring = ConsistentHash()
    hash1 = hash_ring._hash("test_key")
    hash2 = hash_ring._hash("test_key")
    assert hash1 == hash2
    different = hash_ring._hash("different_key")
    assert different != hash1


def test_ring_order_maintained() -> None:
    """Test that sorted_keys is properly maintained."""
    hash_ring = ConsistentHash(replicas=5)
    hash_ring.add_server("server1")
    assert hash_ring.sorted_keys == sorted(hash_ring.sorted_keys)
    hash_ring.add_server("server2")
    assert hash_ring.sorted_keys == sorted(hash_ring.sorted_keys)
    hash_ring.remove_server("server1")
    assert hash_ring.sorted_keys == sorted(hash_ring.sorted_keys)
