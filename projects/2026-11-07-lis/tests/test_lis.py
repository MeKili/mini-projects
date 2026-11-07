"""Tests for longest increasing subsequence solvers."""

from lis import lis_length, lis_subsequence, lis_with_key


class TestLisLength:
    """Test LIS length calculation."""

    def test_empty_sequence(self) -> None:
        assert lis_length([]) == 0

    def test_single_element(self) -> None:
        assert lis_length([5]) == 1

    def test_increasing_sequence(self) -> None:
        assert lis_length([1, 2, 3, 4, 5]) == 5

    def test_decreasing_sequence(self) -> None:
        assert lis_length([5, 4, 3, 2, 1]) == 1

    def test_mixed_sequence(self) -> None:
        assert lis_length([3, 1, 4, 1, 5, 9, 2, 6]) == 4

    def test_with_duplicates(self) -> None:
        assert lis_length([1, 2, 2, 3, 4, 4, 5]) == 5

    def test_all_same(self) -> None:
        assert lis_length([5, 5, 5, 5]) == 1

    def test_two_elements_increasing(self) -> None:
        assert lis_length([1, 2]) == 2

    def test_two_elements_decreasing(self) -> None:
        assert lis_length([2, 1]) == 1

    def test_large_sequence(self) -> None:
        seq = list(range(1000, 0, -1))
        assert lis_length(seq) == 1

    def test_zigzag_pattern(self) -> None:
        assert lis_length([1, 3, 2, 4, 3, 5]) == 4


class TestLisSubsequence:
    """Test LIS reconstruction."""

    def test_empty_sequence(self) -> None:
        assert lis_subsequence([]) == []

    def test_single_element(self) -> None:
        assert lis_subsequence([5]) == [5]

    def test_increasing_sequence(self) -> None:
        result = lis_subsequence([1, 2, 3, 4, 5])
        assert result == [1, 2, 3, 4, 5]

    def test_decreasing_sequence(self) -> None:
        result = lis_subsequence([5, 4, 3, 2, 1])
        assert len(result) == 1

    def test_mixed_sequence(self) -> None:
        result = lis_subsequence([3, 1, 4, 1, 5, 9, 2, 6])
        assert len(result) == 4
        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

    def test_result_is_increasing(self) -> None:
        seq = [1, 5, 0, 4, 1, 3]
        result = lis_subsequence(seq)
        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

    def test_result_length_matches(self) -> None:
        seq = [3, 1, 4, 1, 5, 9, 2, 6]
        length = lis_length(seq)
        result = lis_subsequence(seq)
        assert len(result) == length

    def test_result_is_subsequence(self) -> None:
        seq = [10, 9, 2, 5, 3, 7, 101, 18]
        result = lis_subsequence(seq)
        j = 0
        for i in range(len(seq)):
            if j < len(result) and seq[i] == result[j]:
                j += 1
        assert j == len(result)

    def test_duplicates_handled(self) -> None:
        result = lis_subsequence([1, 2, 2, 3, 4, 4, 5])
        assert len(result) == 5
        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

    def test_negative_numbers(self) -> None:
        result = lis_subsequence([-3, -1, -2, 0, 1, 2])
        assert len(result) == 5
        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

    def test_large_numbers(self) -> None:
        seq = [1000000, 1, 999999, 2, 999998]
        result = lis_subsequence(seq)
        assert len(result) == 3
        assert result == [1, 2, 999998]


class TestLisWithKey:
    """Test LIS with key function."""

    def test_empty_sequence(self) -> None:
        assert lis_with_key([], lambda x: x) == []

    def test_tuple_by_first_element(self) -> None:
        seq = [(1, "a"), (2, "b"), (3, "c"), (4, "d")]
        result = lis_with_key(seq, lambda x: x[0])
        assert result == seq

    def test_tuple_by_first_element_mixed(self) -> None:
        seq = [(3, "a"), (1, "b"), (4, "c"), (1, "d"), (5, "e")]
        result = lis_with_key(seq, lambda x: x[0])
        assert len(result) == 3
        assert all(result[i][0] < result[i + 1][0] for i in range(len(result) - 1))

    def test_string_by_length(self) -> None:
        seq = ["a", "bb", "ccc", "dd", "eeeee"]
        result = lis_with_key(seq, lambda x: len(x))
        assert len(result) == 4
        assert all(len(result[i]) < len(result[i + 1]) for i in range(len(result) - 1))

    def test_custom_class_attribute(self) -> None:
        class Item:
            def __init__(self, value: int) -> None:
                self.value = value

        items = [Item(3), Item(1), Item(4), Item(1), Item(5)]
        result = lis_with_key(items, lambda x: x.value)
        assert len(result) == 3
        assert all(result[i].value < result[i + 1].value for i in range(len(result) - 1))

    def test_absolute_value_key(self) -> None:
        seq = [-1, 2, -3, 4, -5]
        result = lis_with_key(seq, lambda x: abs(x))
        assert len(result) == 5
        assert all(abs(result[i]) < abs(result[i + 1]) for i in range(len(result) - 1))
