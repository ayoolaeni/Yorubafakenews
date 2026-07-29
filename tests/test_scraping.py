from src.scraping.base_scraper import (
    extract_links,
    scrape_fake_article,
    scrape_genuine_article,
    slugify,
)


def test_slugify():
    assert slugify("BBC News Yoruba") == "bbc_news_yoruba"
    assert slugify("Africa Check!") == "africa_check"


def test_extract_links_resolves_relative_and_dedups():
    html = """
    <html><body>
      <article><a href="/story-1">One</a></article>
      <a href="/story-1">Duplicate</a>
      <a href="https://other.example/story-2">Two</a>
      <a href="mailto:x@example.com">Not a link</a>
    </body></html>
    """
    links = extract_links(html, "https://site.example/news", "article a, a")
    assert links == ["https://site.example/story-1", "https://other.example/story-2"]


def test_scrape_genuine_article_extracts_title_and_body():
    html = """
    <html><body>
      <article>
        <h1>Akole Iro</h1>
        <p>Ìpínrọ̀ kìíní.</p>
        <p>Ìpínrọ̀ kejì.</p>
      </article>
    </body></html>
    """
    source_cfg = {
        "name": "Test Source",
        "article_selector": "article",
        "title_selector": "h1",
        "body_selector": "p",
    }
    record = scrape_genuine_article("https://site.example/a", html, source_cfg)
    assert record is not None
    assert record["label"] == "genuine"
    assert record["title"] == "Akole Iro"
    assert "Ìpínrọ̀ kìíní." in record["text"]
    assert "Ìpínrọ̀ kejì." in record["text"]


def test_scrape_genuine_article_returns_none_without_body():
    html = "<html><body><article><h1>Title only</h1></article></body></html>"
    source_cfg = {
        "name": "Test Source",
        "article_selector": "article",
        "title_selector": "h1",
        "body_selector": "p",
    }
    assert scrape_genuine_article("https://site.example/a", html, source_cfg) is None


def test_scrape_fake_article_extracts_claim_and_verdict():
    html = """
    <html><head><title>Fact-check headline</title></head><body>
      <blockquote>The false claim being checked.</blockquote>
      <div class="entry-content"><p>Our verdict: this is false.</p></div>
    </body></html>
    """
    source_cfg = {
        "name": "Test FactCheck",
        "claim_selector": "blockquote",
        "verdict_selector": "div.entry-content p",
    }
    record = scrape_fake_article("https://site.example/claim", html, source_cfg)
    assert record is not None
    assert record["label"] == "fake"
    assert record["text"] == "The false claim being checked."
    assert "verdict" in record["verdict_text"].lower()


def test_scrape_fake_article_matches_claim_keyword_regardless_of_diacritics():
    html = """
    <html><body>
      <div class="entry-content">
        <p>Àhèsọ: ewé mọ̀ríńgà le wo àìsàn ìtọ̀ súgà.</p>
        <p>Àbájáde: èyí kò tọ́ nítorí kò sí ìdánilójú sáyẹ́nsì.</p>
      </div>
    </body></html>
    """
    source_cfg = {
        "name": "Dubawa",
        "claim_selector": "div.entry-content p",
        "claim_keyword": "aheso",
        "verdict_selector": "div.entry-content p",
    }
    record = scrape_fake_article("https://dubawa.org/claim", html, source_cfg)
    assert record is not None
    assert record["text"].startswith("Àhèsọ")


def test_scrape_fake_article_returns_none_without_claim():
    html = "<html><body><div class='entry-content'><p>Just a verdict.</p></div></body></html>"
    source_cfg = {
        "name": "Test FactCheck",
        "claim_selector": "blockquote",
        "verdict_selector": "div.entry-content p",
    }
    assert scrape_fake_article("https://site.example/claim", html, source_cfg) is None