import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import requests
from app.main import app
from app.nli import NLIScorer
from app.retrieval import RetrievalError, search_europe_pmc, extract_fulltext_sections


class MockApiScorer(NLIScorer):
    def score(self, pairs):
        # Deterministic entailing score for synthesis, non-entailing for others
        out = []
        for premise, hyp in pairs:
            if "synthesis" in premise.lower() and "reagent" not in hyp.lower():
                out.append((0.95, 0.04, 0.01))
            elif "reagent" in hyp.lower() and "administered" in premise.lower():
                out.append((0.90, 0.05, 0.05))
            else:
                out.append((0.10, 0.10, 0.80))
        return out


@pytest.fixture(autouse=True)
def inject_mock_scorer():
    """Ensure tests run with fast mock scorer without downloading weights."""
    app.state.scorer = MockApiScorer()


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data


def test_classify_success():
    client = TestClient(app)

    fake_pmc_response = {
        "hitCount": 1,
        "resultList": {
            "result": [
                {
                    "title": "Novel chemical synthesis of psilocybin.",
                    "abstractText": "Here we describe a high-yield chemical synthesis of psilocybin.",
                    "doi": "10.1000/182",
                    "pmid": "12345678",
                    "pubYear": "2024"
                }
            ]
        }
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_pmc_response

        payload = {
            "entity": "psilocybin",
            "criterion": "Psilocybin is synthesised or biosynthesised in this work",
            "synonyms": ["psilocybine"],
            "max_results": 5
        }
        resp = client.post("/classify", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_retrieved"] == 1
        assert len(body["results"]) == 1
        paper = body["results"][0]
        assert paper["classification"] == "qualifies"
        assert paper["evidence_status"] == "entailing_sentence"
        assert "synthesis of psilocybin" in paper["evidence"]
        assert paper["pmid"] == "12345678"


def test_classify_zero_results():
    client = TestClient(app)

    fake_pmc_response = {
        "hitCount": 0,
        "resultList": {"result": []}
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_pmc_response

        payload = {
            "entity": "nonexistentchemical123456",
            "criterion": "Compound is synthesized",
            "max_results": 5
        }
        resp = client.post("/classify", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_retrieved"] == 0
        assert body["results"] == []
        assert "No publications found" in body["note"]


def test_classify_retrieval_failure_502():
    client = TestClient(app)

    with patch("requests.get", side_effect=requests.RequestException("EBI server timeout")):
        payload = {
            "entity": "psilocybin",
            "criterion": "Psilocybin is synthesized",
            "max_results": 5
        }
        resp = client.post("/classify", json=payload)
        assert resp.status_code == 502
        assert "Europe PMC service error" in resp.json()["detail"]


def test_classify_validation_error():
    client = TestClient(app)

    # Empty entity should fail validation
    resp = client.post("/classify", json={
        "entity": "   ",
        "criterion": "Valid criterion",
        "max_results": 10
    })
    assert resp.status_code == 422

    # max_results out of bounds
    resp = client.post("/classify", json={
        "entity": "psilocybin",
        "criterion": "Valid criterion",
        "max_results": 100
    })
    assert resp.status_code == 422


def test_feedback_endpoint(tmp_path):
    client = TestClient(app)
    feedback_payload = {
        "paper_id": "12345678",
        "criterion": "Psilocybin is synthesized",
        "predicted": "does_not_qualify",
        "user_label": "qualifies",
        "corrected_evidence": "We synthesized psilocybin in 4 steps."
    }
    resp = client.post("/feedback", json=feedback_payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_retrieval_retry_on_429():
    """Verify Europe PMC retry logic handles 429 and succeeds on subsequent attempt."""
    mock_response_429 = requests.Response()
    mock_response_429.status_code = 429
    mock_response_429._content = b"Rate limit exceeded"

    mock_response_200 = requests.Response()
    mock_response_200.status_code = 200
    mock_response_200._content = b'{"resultList": {"result": [{"title": "Success paper", "pmid": "123"}]}}'

    with patch("requests.get", side_effect=[mock_response_429, mock_response_200]):
        with patch("time.sleep") as mock_sleep:
            pubs = search_europe_pmc(query='TITLE_ABS:"psilocybin"', max_results=5)
            assert len(pubs) == 1
            assert pubs[0]["title"] == "Success paper"
            assert mock_sleep.called


def test_fulltext_xml_extraction():
    """Verify extraction of text paragraphs from open-access XML."""
    sample_xml = """<article>
        <body>
            <sec sec-type="methods">
                <title>Methods</title>
                <p>Psilocybin was synthesized via phosphorylation of 4-hydroxy-DMT in 85% yield.</p>
            </sec>
        </body>
    </article>"""
    sections = extract_fulltext_sections(sample_xml)
    assert len(sections) == 1
    assert "phosphorylation" in sections[0]


