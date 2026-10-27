def test_empty_note_rejected(client):
    assert client.post("/notes", json={"text": "   "}).status_code == 422


def test_first_user_note_is_n013_and_survives_reload(client):
    n = client.post("/notes", json={"text": "My train to Pune leaves at 6 AM"}).json()
    assert n["id"] == "N013"           # never N001-N012
    # a "restart": a brand-new request reads the same database file
    notes = client.get("/notes").json()
    assert [x["id"] for x in notes] == ["N013"]
    assert notes[0]["category"] == n["category"] and notes[0]["uncertain"] == n["uncertain"]


def test_starter_loads_twice_without_duplicates(client):
    first = client.post("/starter/load").json()
    second = client.post("/starter/load").json()
    assert first["added"] == 12 and second["added"] == 0
    ids = [n["id"] for n in client.get("/notes").json()]
    assert ids == [f"N{i:03d}" for i in range(1, 13)]
    assert client.post("/notes", json={"text": "Buy rice"}).json()["id"] == "N013"


def test_filter_by_category_and_uncertain(client):
    client.post("/starter/load")
    study = client.get("/notes", params={"category": "Study"}).json()
    assert study and all(n["category"] == "Study" for n in study)
    assert client.get("/notes", params={"category": "Nonsense"}).status_code == 400


def test_search_finds_saved_note_and_reports_no_match(client):
    client.post("/starter/load")
    hits = client.get("/search", params={"q": "hotel"}).json()["results"]
    assert hits and hits[0]["id"] == "N007"
    nothing = client.get("/search", params={"q": "zzzzqq"}).json()
    assert nothing["results"] == [] and nothing["message"]
