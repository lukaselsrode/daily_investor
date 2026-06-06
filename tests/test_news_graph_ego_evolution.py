import json

import pandas as pd

from portfolio.visualization import news_graph


def _write_news_snapshot(tmp_path, date, rows):
    path = tmp_path / f"news_{date}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _article(title, link, related=None, publisher="TestWire"):
    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "pub_date": "2026-01-01T12:00:00Z",
        "formatted_date": "01-01-2026",
        "related_symbols": related or [],
    }


def test_ego_network_evolution_walks_dates_and_marks_new_nodes(monkeypatch, tmp_path):
    monkeypatch.setattr(news_graph, "_data_dir", lambda: tmp_path)
    shared_ab = _article("AAA and BBB rally together", "https://example.com/ab")
    shared_ac = _article("AAA and CCC expand partnership", "https://example.com/ac")
    _write_news_snapshot(tmp_path, "2026_01_01", [
        {"symbol": "AAA", "news": json.dumps([shared_ab])},
        {"symbol": "BBB", "news": json.dumps([shared_ab])},
    ])
    _write_news_snapshot(tmp_path, "2026_01_02", [
        {"symbol": "AAA", "news": json.dumps([shared_ab, shared_ac])},
        {"symbol": "BBB", "news": json.dumps([shared_ab])},
        {"symbol": "CCC", "news": json.dumps([shared_ac])},
    ])

    summary, edges, nodes = news_graph.ego_network_evolution("AAA", hops=1)

    assert summary["date"].tolist() == ["2026_01_01", "2026_01_02"]
    assert summary["n_neighbors"].tolist() == [1, 2]
    assert summary.loc[summary["date"] == "2026_01_02", "new_nodes"].item() == "CCC"
    assert set(edges["date"]) == {"2026_01_01", "2026_01_02"}
    assert set(nodes.loc[nodes["date"] == "2026_01_02", "symbol"]) == {"AAA", "BBB", "CCC"}
    assert nodes.loc[(nodes["date"] == "2026_01_01") & (nodes["symbol"] == "AAA"), "is_ego"].item() is True


def test_ego_articles_for_date_links_displayed_neighbors(monkeypatch, tmp_path):
    monkeypatch.setattr(news_graph, "_data_dir", lambda: tmp_path)
    shared_ab = _article("AAA and BBB rally together", "https://example.com/ab")
    structured = _article("AAA supplier mentions DDD", "https://example.com/ad", related=["DDD"])
    _write_news_snapshot(tmp_path, "2026_01_01", [
        {"symbol": "AAA", "news": json.dumps([shared_ab, structured])},
        {"symbol": "BBB", "news": json.dumps([shared_ab])},
    ])

    articles = news_graph.ego_articles_for_date("2026_01_01", "AAA", {"BBB", "DDD"})

    assert set(articles["matched_neighbors"]) == {"BBB", "DDD"}
    assert articles["link"].str.contains("example.com").all()
    assert articles["symbols"].str.contains("AAA").all()
