from fastapi.testclient import TestClient
from cardlens_core.main import app
from cardlens_core.catalog import now, save_card
from test_flow import FIXTURE


def test_catalog_details_include_local_art_and_optional_reference_price():
    save_card(FIXTURE, now())
    client = TestClient(app)
    client.post('/api/v1/session').raise_for_status()
    response = client.get('/api/v1/shop/catalog?q=Test&limit=12').json()
    card = response['items'][0]
    assert card['image'] == '/api/v1/cards/fixture:007:en/image'
    assert card['reference_price']['variant'] == 'normal-unlimited'
    assert card['reference_price']['amount_vnd'] is None
    assert card['reference_price']['reason'] == 'missing_fx'


def test_seller_listings_are_private_and_include_drafts():
    from test_market import account
    save_card(FIXTURE, now())
    seller, _ = account('pkmlens-seller@example.com')
    other, _ = account('pkmlens-other@example.com')
    response = seller.post('/api/v1/listings/manual', json={
        'card_id': 'fixture:007:en', 'variant': 'normal-unlimited',
        'condition': 'NM', 'asking_vnd': 100000,
    })
    response.raise_for_status()
    rows = seller.get('/api/v1/shop/mine').json()
    assert len(rows) == 1 and rows[0]['status'] == 'draft'
    assert other.get('/api/v1/shop/mine').json() == []


def test_saved_catalog_ids_can_be_retrieved_outside_first_page():
    from copy import deepcopy
    for number in range(65):
        card = deepcopy(FIXTURE)
        card.update(id=f'fixture-{number:03}', localId=f'{number:03}')
        save_card(card, now())
    result = TestClient(app).get('/api/v1/shop/catalog?ids=fixture:064:en&limit=60').json()
    assert result['total'] == 1
    assert result['items'][0]['id'] == 'fixture:064:en'


def test_catalog_rarity_filter_runs_before_pagination():
    from copy import deepcopy
    rare = deepcopy(FIXTURE)
    rare.update(id='fixture-008', localId='008', rarity='Illustration Rare')
    save_card(FIXTURE, now()); save_card(rare, now())
    result = TestClient(app).get('/api/v1/shop/catalog?rare=true&limit=1').json()
    assert result['total'] == 1 and result['items'][0]['id'] == 'fixture:008:en'


def test_catalog_sort_order_is_applied_before_page_limits():
    from copy import deepcopy
    for number, name in enumerate(['Zubat', 'Abra', 'Meowth', 'Pikachu']):
        card = deepcopy(FIXTURE)
        card.update(id=f'fixture-{number:03}', localId=f'{number:03}', name=name)
        save_card(card, now())
    client = TestClient(app)
    result = client.get('/api/v1/shop/catalog?limit=2&sort=name').json()
    assert [card['name'] for card in result['items']] == ['Abra', 'Meowth']


def test_catalog_price_sort_includes_cards_beyond_first_page():
    from copy import deepcopy
    from cardlens_core.db import Fx, Session
    from datetime import datetime, timezone
    for number, (name, amount) in enumerate([('Abra', 10), ('Zubat', 500), ('Meowth', 30)]):
        card = deepcopy(FIXTURE)
        card.update(id=f'fixture-{number:03}', localId=f'{number:03}', name=name)
        variant = card['variants_detailed'][0]
        variant['thirdParty'] = {'tcgplayer': 9000 + number}
        variant['pricing']['tcgplayer']['normal']['marketPrice'] = amount
        save_card(card, now())
    with Session.begin() as db:
        db.add(Fx(id='fx', source_at=str(int(datetime.now(timezone.utc).timestamp())), fetched_at=now(), source='fixture', rates={'USD': '1', 'VND': '25000'}))
    result = TestClient(app).get('/api/v1/shop/catalog?limit=1&sort=price-down').json()
    assert result['items'][0]['name'] == 'Zubat'
