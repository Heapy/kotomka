from kotomka.utils import format_timecode, parse_showinfo_timestamps


def test_format_timecode() -> None:
    assert format_timecode(0) == "00:00"
    assert format_timecode(65) == "01:05"
    assert format_timecode(3661) == "01:01:01"


def test_parse_showinfo_timestamps() -> None:
    stderr = "n:0 pts_time:1.25 foo\nn:1 pts_time:9 bar"
    assert parse_showinfo_timestamps(stderr) == [1.25, 9.0]

