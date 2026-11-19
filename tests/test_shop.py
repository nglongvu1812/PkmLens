from fastapi.testclient import TestClient
from cardlens_core.main import app
from cardlens_core.catalog import now, save_card
from test_flow import FIXTURE


def test_catalog_browse_and_batch_quotes_keep_variants_explicit():
    save_card(FIXTURE, now())
    client = TestClient(app)
    client.post('/api/v1/session').raise_for_status()
    response = client.get('/api/v1/shop/catalog?q=Test&limit=12')
    assert response.status_code == 200
    body = response.json()
    assert body['total'] == 1
    assert body['items'][0]['id'] == 'fixture:007:en'
    quotes = client.post('/api/v1/prices/batch', json={'items': [
        {'card_id': 'fixture:007:en', 'variant': 'normal-unlimited', 'condition': 'NM'},
        {'card_id': 'fixture:007:en', 'variant': 'not-a-variant', 'condition': 'NM'},
    ]})
    assert quotes.status_code == 422
    assert client.get('/api/v1/shop/cards/fixture:007:en').json()['card']['number'] == '007'


def test_scan_record_persists_scan_and_live_online_price(monkeypatch):
    from cardlens_core import live_price
    from cardlens_core.db import Fx, Session
    save_card({**FIXTURE, "id": "fixture-live-9", "localId": "9", "name": "Livechu", "variants_detailed": [{"variantId": "normal", "type": "normal"}]}, now())
    with Session.begin() as db:
        db.merge(Fx(id="fx-test", source_at="1999999999", fetched_at=now(), source="test-fx", rates={"USD": "1", "VND": "25000"}))

    class FakeResp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None):
            if url.endswith('/cards'):
                return FakeResp([{'id': 'sv03-9', 'localId': '9', 'name': 'Livechu'}])
            return FakeResp({
                'id': 'sv03-9', 'localId': '9', 'name': 'Livechu',
                'set': {'name': 'Fixture'},
                'pricing': {'tcgplayer': {'unit': 'USD', 'updated': now(), 'normal': {'productId': 12345, 'marketPrice': 4.0, 'lowPrice': 3.0, 'highPrice': 6.0}}},
            })

    monkeypatch.setenv("PKMLENS_LIVE_PRICE_FETCH", "1")
    monkeypatch.setattr(live_price.httpx, "Client", FakeClient)
    live_price._recent_attempts.clear()

    client = TestClient(app)
    client.post('/api/v1/session').raise_for_status()
    recorded = client.post('/api/v1/scan/record', json={'card_id': 'fixture:9:en', 'similarity': 0.88, 'source': 'camera', 'auto_added': True})
    assert recorded.status_code == 200
    payload = recorded.json()
    assert payload['price_fetched'] is True
    assert payload['quote']['amount_vnd'] == 100000
    assert payload['quote']['source'] == 'TCGdex/tcgplayer'
