import pytest
from app.nli import DebertaNLIScorer
from app.classifier import classify_publications


@pytest.fixture(scope="module")
def real_scorer():
    """Load real deberta-v3-small model once for slow integration tests."""
    return DebertaNLIScorer()


@pytest.mark.slow
def test_real_model_synthesis_qualifies(real_scorer):
    """Assertion 1: 'We report a total synthesis of psilocybin in three steps' -> qualifies."""
    sentence = "We report a total synthesis of psilocybin in three steps."
    publications = [{
        "title": sentence,
        "abstractText": sentence,
        "pmid": "syn_01"
    }]
    results, _, _, _ = classify_publications(
        publications=publications,
        entity="psilocybin",
        criterion="Psilocybin was synthesized or biosynthesized in this work.",
        threshold=0.40,
        contrast_delta=0.20,
        scorer=real_scorer
    )
    assert results[0]["classification"] == "qualifies"
    assert results[0]["evidence_status"] == "entailing_sentence"
    assert results[0]["qualify_score"] > 0.80


@pytest.mark.slow
def test_real_model_administered_does_not_qualify(real_scorer):
    """Assertion 2: 'Psilocybin was administered to participants in a randomised trial' -> does_not_qualify."""
    sentence = "Psilocybin was administered to participants in a randomised trial."
    publications = [{
        "title": sentence,
        "abstractText": sentence,
        "pmid": "adm_01"
    }]
    results, _, _, _ = classify_publications(
        publications=publications,
        entity="psilocybin",
        criterion="Psilocybin was synthesized or biosynthesized in this work.",
        threshold=0.40,
        contrast_delta=0.20,
        scorer=real_scorer
    )
    assert results[0]["classification"] == "does_not_qualify"


@pytest.mark.slow
def test_real_model_starting_material_does_not_qualify(real_scorer):
    """
    Assertion 3: 'Compound 4 was prepared using psilocybin as a starting material' -> does_not_qualify.
    With multi-hypothesis affirmative contrast guard, reactant use is recognized and guard fires.
    """
    sentence = "Compound 4 was prepared using psilocybin as a starting material."
    publications = [{
        "title": sentence,
        "abstractText": sentence,
        "pmid": "start_01"
    }]
    results, _, _, _ = classify_publications(
        publications=publications,
        entity="psilocybin",
        criterion="Psilocybin was synthesized or biosynthesized in this work.",
        threshold=0.40,
        scorer=real_scorer
    )
    assert results[0]["classification"] == "does_not_qualify"
    assert results[0]["evidence_status"] == "contrast_mention"


@pytest.mark.slow
def test_real_model_negation_case(real_scorer):
    """
    Assertion 4: 'No synthesis of psilocybin was observed' -> does_not_qualify.
    Documented known-error test for NLI negation sensitivity.
    """
    sentence = "No synthesis of psilocybin was observed."
    publications = [{
        "title": sentence,
        "abstractText": sentence,
        "pmid": "neg_01"
    }]
    results, _, _, _ = classify_publications(
        publications=publications,
        entity="psilocybin",
        criterion="Psilocybin was synthesized or biosynthesized in this work.",
        threshold=0.40,
        contrast_delta=0.20,
        scorer=real_scorer
    )
    assert results[0]["classification"] == "does_not_qualify"

