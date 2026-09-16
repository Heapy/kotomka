from kotomka.app import _citation_links


def test_citation_links_turn_seconds_into_timecodes() -> None:
    html = str(_citation_links("See [t=58, 120, 163].", "https://example.com/watch?v=1"))

    assert "00:58" in html
    assert "02:00" in html
    assert "02:43" in html
    assert "t=58s" in html


def test_citation_links_escapes_text() -> None:
    html = str(_citation_links("<script>x</script> [1]", "https://example.com"))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_only_explicit_citations_outside_code_become_links() -> None:
    text = "Numbers [16, 32, 64]. `data[t=45]`\n```\n[t=45]\n```\nSee [t=45]."
    html = str(_citation_links(text, "https://example.com"))
    assert "[16, 32, 64]" in html
    assert "`data[t=45]`" in html
    assert "```\n[t=45]\n```" in html
    assert html.count('class="time-link"') == 1
    assert "00:45" in html


def test_pdf_citations_use_readable_times_and_preserve_metrics() -> None:
    from kotomka.pdf import _report_text
    assert _report_text("Sizes [16, 32, 64], see [t=45].") == "Sizes [16, 32, 64], see [00:45]."
