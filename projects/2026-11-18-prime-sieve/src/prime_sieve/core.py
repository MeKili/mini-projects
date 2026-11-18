"""Sieve of Eratosthenes for efficient prime enumeration and primality testing."""


def sieve(n: int) -> list[int]:
    """Generate all primes up to n using the Sieve of Eratosthenes.

    Args:
        n: Upper bound (inclusive).

    Returns:
        List of all primes <= n, sorted.

    Raises:
        ValueError: If n < 2.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]


def is_prime(n: int) -> bool:
    """Check if n is prime using trial division.

    Args:
        n: Number to test.

    Returns:
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    return all(n % i != 0 for i in range(3, int(n**0.5) + 1, 2))


def prime_factors(n: int) -> list[int]:
    """Return prime factorization of n as a list of factors (with repetition).

    Args:
        n: Number to factorize.

    Returns:
        List of prime factors in ascending order.

    Raises:
        ValueError: If n < 2.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def nth_prime(n: int) -> int:
    """Return the nth prime (1-indexed).

    Args:
        n: Position (1 = first prime = 2).

    Returns:
        The nth prime number.

    Raises:
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    if n == 1:
        return 2

    count = 1
    candidate = 3
    while count < n:
        if is_prime(candidate):
            count += 1
        if count < n:
            candidate += 2
    return candidate


def twin_primes(limit: int) -> list[tuple[int, int]]:
    """Find all twin prime pairs (p, p+2) up to limit.

    Args:
        limit: Upper bound (inclusive).

    Returns:
        List of (p, p+2) tuples where both are prime.

    Raises:
        ValueError: If limit < 2.
    """
    if limit < 2:
        raise ValueError(f"limit must be >= 2, got {limit}")

    primes_set = set(sieve(limit))
    return [(p, p + 2) for p in primes_set if p + 2 in primes_set]
