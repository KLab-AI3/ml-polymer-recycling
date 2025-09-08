"""Tests for multi-format file parsing functionality."""

import pytest
import numpy as np
from utils.multifile import (
    parse_spectrum_data,
    detect_file_format,
    parse_json_spectrum,
    parse_csv_spectrum,
    parse_txt_spectrum,
)


def test_detect_file_format():
    """Test automatic file format detection."""
    # JSON detection
    json_content = '{"wavenumbers": [1, 2, 3], "intensities": [0.1, 0.2, 0.3]}'
    assert detect_file_format("test.json", json_content) == "json"

    # CSV detection
    csv_content = "wavenumber,intensity\n1000,0.5\n1001,0.6"
    assert detect_file_format("test.csv", csv_content) == "csv"

    # TXT detection (default)
    txt_content = "1000 0.5\n1001 0.6"
    assert detect_file_format("test.txt", txt_content) == "txt"


def test_parse_json_spectrum():
    """Test JSON spectrum parsing."""
    # Test object format
    json_content = '{"wavenumbers": [1000, 1001, 1002], "intensities": [0.1, 0.2, 0.3]}'
    x, y = parse_json_spectrum(json_content)

    expected_x = np.array([1000, 1001, 1002])
    expected_y = np.array([0.1, 0.2, 0.3])

    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)

    # Test alternative key names
    json_content_alt = '{"x": [1000, 1001, 1002], "y": [0.1, 0.2, 0.3]}'
    x_alt, y_alt = parse_json_spectrum(json_content_alt)
    np.testing.assert_array_equal(x_alt, expected_x)
    np.testing.assert_array_equal(y_alt, expected_y)

    # Test array of objects format
    json_array = """[
        {"wavenumber": 1000, "intensity": 0.1},
        {"wavenumber": 1001, "intensity": 0.2},
        {"wavenumber": 1002, "intensity": 0.3}
    ]"""
    x_arr, y_arr = parse_json_spectrum(json_array)
    np.testing.assert_array_equal(x_arr, expected_x)
    np.testing.assert_array_equal(y_arr, expected_y)


def test_parse_csv_spectrum():
    """Test CSV spectrum parsing."""
    # Test with headers
    csv_with_headers = """wavenumber,intensity
1000,0.1
1001,0.2
1002,0.3
1003,0.4
1004,0.5
1005,0.6
1006,0.7
1007,0.8
1008,0.9
1009,1.0
1010,1.1
1011,1.2"""

    x, y = parse_csv_spectrum(csv_with_headers)
    expected_x = np.array(
        [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011]
    )
    expected_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)

    # Test without headers
    csv_no_headers = """1000,0.1
1001,0.2
1002,0.3
1003,0.4
1004,0.5
1005,0.6
1006,0.7
1007,0.8
1008,0.9
1009,1.0
1010,1.1
1011,1.2"""

    x_no_h, y_no_h = parse_csv_spectrum(csv_no_headers)
    np.testing.assert_array_equal(x_no_h, expected_x)
    np.testing.assert_array_equal(y_no_h, expected_y)

    # Test semicolon delimiter
    csv_semicolon = """1000;0.1
1001;0.2
1002;0.3
1003;0.4
1004;0.5
1005;0.6
1006;0.7
1007;0.8
1008;0.9
1009;1.0
1010;1.1
1011;1.2"""

    x_semi, y_semi = parse_csv_spectrum(csv_semicolon)
    np.testing.assert_array_equal(x_semi, expected_x)
    np.testing.assert_array_equal(y_semi, expected_y)


def test_parse_txt_spectrum():
    """Test TXT spectrum parsing."""
    txt_content = """# Comment line
1000 0.1
1001 0.2
1002 0.3
1003 0.4
1004 0.5
1005 0.6
1006 0.7
1007 0.8
1008 0.9
1009 1.0
1010 1.1
1011 1.2"""

    x, y = parse_txt_spectrum(txt_content)
    expected_x = np.array(
        [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011]
    )
    expected_y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, expected_y)

    # Test comma-separated
    txt_comma = """1000,0.1
1001,0.2
1002,0.3
1003,0.4
1004,0.5
1005,0.6
1006,0.7
1007,0.8
1008,0.9
1009,1.0
1010,1.1
1011,1.2"""

    x_comma, y_comma = parse_txt_spectrum(txt_comma)
    np.testing.assert_array_equal(x_comma, expected_x)
    np.testing.assert_array_equal(y_comma, expected_y)


def test_parse_spectrum_data_integration():
    """Test integrated spectrum data parsing with format detection."""
    # Test automatic format detection and parsing
    test_cases = [
        (
            '{"wavenumbers": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011], "intensities": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]}',
            "test.json",
        ),
        (
            "wavenumber,intensity\n1000,0.1\n1001,0.2\n1002,0.3\n1003,0.4\n1004,0.5\n1005,0.6\n1006,0.7\n1007,0.8\n1008,0.9\n1009,1.0\n1010,1.1\n1011,1.2",
            "test.csv",
        ),
        (
            "1000 0.1\n1001 0.2\n1002 0.3\n1003 0.4\n1004 0.5\n1005 0.6\n1006 0.7\n1007 0.8\n1008 0.9\n1009 1.0\n1010 1.1\n1011 1.2",
            "test.txt",
        ),
    ]

    for content, filename in test_cases:
        x, y = parse_spectrum_data(content, filename)
        assert len(x) >= 10
        assert len(y) >= 10
        assert len(x) == len(y)


def test_insufficient_data_points():
    """Test handling of insufficient data points."""
    # Test with too few points
    insufficient_data = "1000 0.1\n1001 0.2"  # Only 2 points, need at least 10

    with pytest.raises(ValueError, match="Insufficient data points"):
        parse_txt_spectrum(insufficient_data, "test.txt")


def test_invalid_json():
    """Test handling of invalid JSON."""
    invalid_json = (
        '{"wavenumbers": [1000, 1001], "intensities": [0.1}'  # Missing closing bracket
    )

    with pytest.raises(ValueError, match="Invalid JSON format"):
        parse_json_spectrum(invalid_json)


def test_empty_file():
    """Test handling of empty files."""
    empty_content = ""

    with pytest.raises(ValueError, match="No data lines found"):
        parse_txt_spectrum(empty_content, "empty.txt")


if __name__ == "__main__":
    pytest.main([__file__])
