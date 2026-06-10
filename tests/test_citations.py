from kotomka.app import _citation_links


def test_citation_links_turn_seconds_into_timecodes() -> None:
    html = str(_citation_links("See [58, 120, 163].", "https://example.com/watch?v=1"))

    assert "00:58" in html
    assert "02:00" in html
    assert "02:43" in html
    assert "t=58s" in html


def test_citation_links_escapes_text() -> None:
    html = str(_citation_links("<script>x</script> [1]", "https://example.com"))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
