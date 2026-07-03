from scholar_agent.planning.query_generation import heuristic_generate_search_queries
from scholar_agent.planning.query_understanding import heuristic_understand_query
from scholar_agent.retrieval.dynamic_router import get_routing_config


def _query_texts(question: str) -> list[str]:
    plan = heuristic_understand_query(question)
    return [item.query.lower() for item in heuristic_generate_search_queries(plan)]


def test_spar_sentiment_question_gets_academic_fallback_terms():
    texts = _query_texts(
        'How can AI methods improve sentiment analysis model accuracy? For instance, '
        '"How is Xiaoming?" might have completely different meanings depending on the context'
    )

    assert any("contextual sentiment analysis" in text for text in texts)
    assert any("aspect-level sentiment classification" in text for text in texts)


def test_spar_legal_llm_question_gets_legal_nlp_terms():
    question = (
        "How can large-scale language models improve automated legal text analysis systems "
        "to minimize human intervention?"
    )
    plan = heuristic_understand_query(question)
    texts = [item.query.lower() for item in heuristic_generate_search_queries(plan)]

    assert "large language model" in plan.methods
    assert any("legal" in entity for entity in plan.entities)
    assert any("legal text analysis" in text for text in texts)
    assert any("legal nlp" in text or "legal natural language processing" in text for text in texts)


def test_spar_occluded_face_question_gets_masked_face_terms():
    question = (
        "How can deep neural networks enhance real-time facial recognition performance while "
        "reducing processing time? If a person is partially occluded, such as wearing a mask, "
        "how can the system still recognize them?"
    )
    plan = heuristic_understand_query(question)
    texts = [item.query.lower() for item in heuristic_generate_search_queries(plan)]

    assert "deep neural network" in plan.methods
    assert "masked face recognition" in plan.entities
    assert "occluded face recognition" in plan.entities
    assert "face recognition under occlusion" in plan.entities
    assert "efficient face recognition" in plan.entities
    assert "real-time face recognition" in plan.entities
    assert any("masked face recognition" in text for text in texts)
    assert any("occluded face recognition" in text for text in texts)
    assert any("efficient face recognition" in text for text in texts)


def test_biomedical_latest_work_routes_pubmed_core_queries():
    plan = heuristic_understand_query(
        "What breakthrough advancements have been made in lung cancer research? "
        "Present the latest developments and challenges in treatment."
    )

    routing = get_routing_config(plan)

    assert plan.query_type == "latest_work"
    assert "lung cancer" in plan.entities
    assert "non-small cell lung cancer" in plan.entities
    assert "core_topic" in routing.routes_per_provider["pubmed"]
    assert routing.caps_per_provider["pubmed"] >= 3


def test_spar_autonomous_driving_question_gets_decision_terms():
    question = (
        "How can autonomous driving systems use deep learning to improve robotic "
        "decision-making in complex traffic scenes?"
    )
    plan = heuristic_understand_query(question)
    texts = [item.query.lower() for item in heuristic_generate_search_queries(plan)]

    assert "autonomous driving" in plan.entities
    assert "autonomous vehicle decision making" in plan.entities
    assert any("autonomous vehicle decision making" in text for text in texts)


def test_spar_alzheimer_csf_question_gets_biomarker_terms():
    question = (
        "How do tau protein and beta-amyloid concentration changes in cerebrospinal "
        "fluid impact the prediction of Alzheimer's disease progression in early diagnosis?"
    )
    plan = heuristic_understand_query(question)
    texts = [item.query.lower() for item in heuristic_generate_search_queries(plan)]

    assert "cerebrospinal fluid biomarkers" in plan.entities
    assert "tau protein" in plan.entities
    assert any("cerebrospinal fluid biomarkers" in text for text in texts)
    assert any("alzheimer disease progression" in text for text in texts)


def test_spar_personalized_immunotherapy_question_gets_precision_oncology_terms():
    question = (
        "How can personalized immunotherapy be optimized for cancer treatment based "
        "on different patient conditions? Can artificial intelligence be integrated?"
    )
    plan = heuristic_understand_query(question)
    texts = [item.query.lower() for item in heuristic_generate_search_queries(plan)]

    assert "personalized immunotherapy" in plan.entities
    assert "precision oncology" in plan.entities
    assert any("personalized cancer immunotherapy" in text for text in texts)
    assert any("immune checkpoint blockade" in text for text in texts)


def test_focused_extension_question_routes_specific_paper():
    plan = heuristic_understand_query(
        "Which papers extended IPS and SNIPS methods to implicit feedback data?"
    )

    assert plan.query_type == "specific_paper"
    assert "IPS" in plan.entities
    assert "SNIPS" in plan.entities
