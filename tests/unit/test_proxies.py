import responses
from algomlb.ingestion.proxies import fetch_free_proxies


@responses.activate
def test_fetch_free_proxies_success():
    responses.add(
        responses.GET,
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
        body="1.1.1.1:80\n2.2.2.2:8080\n",
        status=200,
    )

    proxies = fetch_free_proxies()
    assert len(proxies) == 2
    assert proxies[0] == "http://1.1.1.1:80"
    assert proxies[1] == "http://2.2.2.2:8080"


@patch("requests.get")
def test_fetch_free_proxies_failure(mock_get):
    mock_get.side_effect = Exception("API Down")
    proxies = fetch_free_proxies()
    assert proxies == []
