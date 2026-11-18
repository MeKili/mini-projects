"""Tests for prime sieve and primality utilities."""

import pytest

from prime_sieve import is_prime, nth_prime, prime_factors, sieve, twin_primes


class TestSieve:
    """Tests for Sieve of Eratosthenes."""

    def test_sieve_small(self) -> None:
        """Test sieve on small range."""
        assert sieve(10) == [2, 3, 5, 7]

    def test_sieve_larger(self) -> None:
        """Test sieve on larger range."""
        primes = sieve(30)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_sieve_two(self) -> None:
        """Test sieve with n=2."""
        assert sieve(2) == [2]

    def test_sieve_one_raises(self) -> None:
        """Test sieve raises on n < 2."""
        with pytest.raises(ValueError):
            sieve(1)

    def test_sieve_zero_raises(self) -> None:
        """Test sieve raises on n = 0."""
        with pytest.raises(ValueError):
            sieve(0)

    def test_sieve_negative_raises(self) -> None:
        """Test sieve raises on negative n."""
        with pytest.raises(ValueError):
            sieve(-5)

    def test_sieve_100(self) -> None:
        """Test sieve on 100."""
        primes = sieve(100)
        assert len(primes) == 25
        assert primes[0] == 2
        assert primes[-1] == 97


class TestIsPrime:
    """Tests for primality testing."""

    def test_is_prime_true(self) -> None:
        """Test is_prime on known primes."""
        assert is_prime(2)
        assert is_prime(3)
        assert is_prime(5)
        assert is_prime(7)
        assert is_prime(11)
        assert is_prime(97)

    def test_is_prime_false(self) -> None:
        """Test is_prime on non-primes."""
        assert not is_prime(1)
        assert not is_prime(0)
        assert not is_prime(4)
        assert not is_prime(6)
        assert not is_prime(9)
        assert not is_prime(100)

    def test_is_prime_negative(self) -> None:
        """Test is_prime on negative numbers."""
        assert not is_prime(-1)
        assert not is_prime(-5)
        assert not is_prime(-100)

    def test_is_prime_large(self) -> None:
        """Test is_prime on large numbers."""
        assert is_prime(10007)
        assert is_prime(104729)
        assert not is_prime(10008)
        assert not is_prime(104730)


class TestPrimeFactors:
    """Tests for prime factorization."""

    def test_prime_factors_simple(self) -> None:
        """Test factorization of simple numbers."""
        assert prime_factors(2) == [2]
        assert prime_factors(3) == [3]
        assert prime_factors(5) == [5]

    def test_prime_factors_composite(self) -> None:
        """Test factorization of composite numbers."""
        assert prime_factors(4) == [2, 2]
        assert prime_factors(6) == [2, 3]
        assert prime_factors(8) == [2, 2, 2]
        assert prime_factors(12) == [2, 2, 3]

    def test_prime_factors_large(self) -> None:
        """Test factorization of larger numbers."""
        assert prime_factors(100) == [2, 2, 5, 5]
        assert prime_factors(123) == [3, 41]
        assert prime_factors(997) == [997]

    def test_prime_factors_one_raises(self) -> None:
        """Test prime_factors raises on 1."""
        with pytest.raises(ValueError):
            prime_factors(1)

    def test_prime_factors_zero_raises(self) -> None:
        """Test prime_factors raises on 0."""
        with pytest.raises(ValueError):
            prime_factors(0)

    def test_prime_factors_negative_raises(self) -> None:
        """Test prime_factors raises on negative."""
        with pytest.raises(ValueError):
            prime_factors(-5)


class TestNthPrime:
    """Tests for nth prime lookup."""

    def test_nth_prime_first_few(self) -> None:
        """Test first few primes."""
        assert nth_prime(1) == 2
        assert nth_prime(2) == 3
        assert nth_prime(3) == 5
        assert nth_prime(4) == 7
        assert nth_prime(5) == 11

    def test_nth_prime_larger(self) -> None:
        """Test larger indices."""
        assert nth_prime(10) == 29
        assert nth_prime(25) == 97

    def test_nth_prime_zero_raises(self) -> None:
        """Test nth_prime raises on 0."""
        with pytest.raises(ValueError):
            nth_prime(0)

    def test_nth_prime_negative_raises(self) -> None:
        """Test nth_prime raises on negative."""
        with pytest.raises(ValueError):
            nth_prime(-1)


class TestTwinPrimes:
    """Tests for twin prime pairs."""

    def test_twin_primes_small(self) -> None:
        """Test twin primes up to 20."""
        pairs = twin_primes(20)
        assert (3, 5) in pairs
        assert (5, 7) in pairs
        assert (11, 13) in pairs
        assert (17, 19) in pairs
        assert len(pairs) == 4

    def test_twin_primes_larger(self) -> None:
        """Test twin primes up to 100."""
        pairs = twin_primes(100)
        expected = [
            (3, 5),
            (5, 7),
            (11, 13),
            (17, 19),
            (29, 31),
            (41, 43),
            (59, 61),
            (71, 73),
        ]
        assert pairs == expected

    def test_twin_primes_exact_boundary(self) -> None:
        """Test boundary condition with twin prime."""
        pairs = twin_primes(5)
        assert (3, 5) in pairs

    def test_twin_primes_one_raises(self) -> None:
        """Test twin_primes raises on limit < 2."""
        with pytest.raises(ValueError):
            twin_primes(1)

    def test_twin_primes_zero_raises(self) -> None:
        """Test twin_primes raises on limit = 0."""
        with pytest.raises(ValueError):
            twin_primes(0)

    def test_twin_primes_negative_raises(self) -> None:
        """Test twin_primes raises on negative limit."""
        with pytest.raises(ValueError):
            twin_primes(-5)
