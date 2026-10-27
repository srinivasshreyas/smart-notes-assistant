import pytest

LABELS = {"Study", "Money", "Travel", "Shopping"}


def test_classify_valid_note(client):
    r = client.post("/classify", json={"text": "Pay the electricity bill tomorrow."})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in LABELS
    assert isinstance(body["uncertain"], bool)


@pytest.mark.parametrize("payload", [
    {"text": ""}, {"text": "   "}, {"note": "Pay rent"}, {"text": 42}, {"text": None}, {"text": ["a"]},
])
def test_classify_invalid_input_is_4xx(client, payload):
    assert 400 <= client.post("/classify", json=payload).status_code < 500


def test_classify_malformed_json_and_server_survives(client):
    r = client.post("/classify", content=b'{"text": ', headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500
    assert client.post("/classify", json={"text": "Book a train to Pune."}).status_code == 200


def test_classify_is_deterministic(client):
    a = client.post("/classify", json={"text": "Buy a new laptop stand."}).json()
    b = client.post("/classify", json={"text": "Buy a new laptop stand."}).json()
    assert a == b


def test_classify_1000_chars_with_rupee_sign(client):
    text = ("Move \u20b92,000 to savings. " * 60)[:1000]
    assert len(text) == 1000
    assert client.post("/classify", json={"text": text}).status_code == 200


def test_out_of_scope_note_is_flagged(client):
    r = client.post("/classify", json={"text": "Call grandma this weekend."}).json()
    assert r["uncertain"] is True
