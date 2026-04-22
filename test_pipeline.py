
import os
import sys
import logging
from rag_pipeline import RAGPipeline

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Ground Truth - small but representive data
# Covers: English, Arabic, answerable, and unanswerable 
GROUND_TRUTH = {
    # English - answerable
    "What is the legally mandated number of annual leave days?": {
        "document": "Employment-Rights-Guide.pdf",
        "pages": [16],
        "answerable": True,
        "language": "en"
    },
    "Are self-employed individuals required to pay municipal fees, and on what basis are they calculated?": {
        "document": "Self-Employment-Guide.pdf",
        "pages": [34],
        "answerable": True,
        "language": "en"
    },
    "What compensation is due in case of abusive dismissal?": {
        "document": "Employment-Rights-Guide.pdf",
        "pages": [21],
        "answerable": True,
        "language": "en"
    },
    # English - unanswerable (refusal expected)
    "Does the guide set minimum pricing standards for freelance services?": {
        "document": None,
        "pages": [],
        "answerable": False,
        "language": "en"
    },
    "Does the guide regulate cryptocurrency salary payments?": {
        "document": None,
        "pages": [],
        "answerable": False,
        "language": "en"
    },
    # Arabic - answerable
    "ما هو العدد القانوني المقرر لأيام الإجازة السنوية؟": {
        "document": "دليل عن حقوق العمل في لبنان.pdf",
        "pages": [16],
        "answerable": True,
        "language": "ar"
    },
    "ما هو التعويض المستحق في حالة الفصل التعسفي؟": {
        "document": "دليل عن حقوق العمل في لبنان.pdf",
        "pages": [21],
        "answerable": True,
        "language": "ar"
    },
    # Arabic - unanswerable (refusal expected)
    "هل ينظم الدليل دفع الرواتب بالعملات المشفرة؟": {
        "document": None,
        "pages": [],
        "answerable": False,
        "language": "ar"
    },
}

questions = list(GROUND_TRUTH.keys())
questions_en = [q for q in questions if GROUND_TRUTH[q]["language"] == "en"]
questions_ar = [q for q in questions if GROUND_TRUTH[q]["language"] == "ar"]


# Retrieval metric helpers

def hit_at_k(pipeline, questions, k=3):
    hits = 0
    total = 0

    for question in questions:
        gt = GROUND_TRUTH[question]
        if not gt["answerable"]:
            continue

        total += 1
        gt_doc = gt["document"]
        gt_pages = set(str(p) for p in gt["pages"])
        retrieved = pipeline.retrieve(question, k=k)

        for doc in retrieved:
            if doc["source"] == gt_doc and str(doc["page"]) in gt_pages:
                hits += 1
                break

    return hits / total if total else 0.0


def mean_reciprocal_rank(pipeline, questions, k=10):
    total = 0
    rr_sum = 0.0

    for question in questions:
        gt = GROUND_TRUTH[question]
        if not gt["answerable"]:
            continue

        total += 1
        gt_doc = gt["document"]
        gt_pages = set(str(p) for p in gt["pages"])
        retrieved = pipeline.retrieve(question, k=k)

        for rank, doc in enumerate(retrieved, start=1):
            if doc["source"] == gt_doc and str(doc["page"]) in gt_pages:
                rr_sum += 1.0 / rank
                break

    return rr_sum / total if total else 0.0


# Test functions - each returns True (pass) or False (fail)
def test_pipeline_loads():
    """Pipeline initializes without errors."""
    logger.info("TEST: pipeline initialization")
    try:
        pipeline = RAGPipeline()
        assert pipeline.index is not None, "FAISS index is None"
        assert len(pipeline.chunks) > 0, "Chunks list is empty"
        assert pipeline.bm25 is not None, "BM25 index is None"
        logger.info("PASS: pipeline loaded — %d chunks, %d vectors",
                    len(pipeline.chunks), pipeline.index.ntotal)
        return pipeline
    except Exception as e:
        logger.error("FAIL: pipeline failed to load — %s", e)
        return None


def test_language_detection(pipeline):
    """Language detection returns correct labels."""
    logger.info("TEST: language detection")
    cases = [
        ("What is the minimum wage?", "en"),
        ("ما هو الحد الأدنى للأجور؟", "ar"),
        ("Hello world", "en"),
        ("مرحبا بالعالم", "ar"),
    ]
    passed = True
    for text, expected in cases:
        result = pipeline._detect_language(text)
        if result != expected:
            logger.error("FAIL: detect_language('%s') = '%s', expected '%s'",
                         text[:40], result, expected)
            passed = False
        else:
            logger.info("PASS: '%s' → %s", text[:40], result)
    return passed


def test_retrieve_returns_results(pipeline):
    """retrieve() returns non-empty results for known answerable questions."""
    logger.info("TEST: retrieve() returns results")
    passed = True
    for question in questions:
        if not GROUND_TRUTH[question]["answerable"]:
            continue
        results = pipeline.retrieve(question, k=5)
        if not results:
            logger.error("FAIL: no results returned for: %s", question[:60])
            passed = False
        else:
            logger.info("PASS: %d results for '%s'", len(results), question[:60])
    return passed


def test_retrieve_language_filter(pipeline):
    """retrieve() only returns chunks matching the query language."""
    logger.info("TEST: language filter in retrieve()")
    passed = True
    for question in questions:
        expected_lang = GROUND_TRUTH[question]["language"]
        results = pipeline.retrieve(question, k=5)
        for doc in results:
            if doc["language"] != expected_lang:
                logger.error(
                    "FAIL: language mismatch — query=%s, chunk_lang=%s, expected=%s",
                    question[:40], doc["language"], expected_lang
                )
                passed = False
    if passed:
        logger.info("PASS: all retrieved chunks match query language")
    return passed


def test_refusal_on_unanswerable(pipeline):
    """generate_answer() refuses unanswerable questions - no hallucination."""
    logger.info("TEST: refusal on unanswerable questions")
    refusal_phrases = ["does not provide", "لا يوفر"]
    passed = True

    unanswerable = [q for q in questions if not GROUND_TRUTH[q]["answerable"]]
    for question in unanswerable:
        result = pipeline.generate_answer(question)
        answer_lower = result["answer"].lower()
        is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
        if not is_refusal:
            logger.error("FAIL: expected refusal but got answer — '%s'", question[:60])
            logger.error("       Answer: %s", result["answer"][:120])
            passed = False
        else:
            logger.info("PASS: correctly refused '%s'", question[:60])
    return passed


def test_answer_has_sources(pipeline):
    """generate_answer() returns sources for answerable questions."""
    logger.info("TEST: sources returned for answerable questions")
    passed = True

    answerable = [q for q in questions_en if GROUND_TRUTH[q]["answerable"]]
    for question in answerable:
        result = pipeline.generate_answer(question)
        refusal_phrases = ["does not provide", "لا يوفر"]
        is_refusal = any(p in result["answer"].lower() for p in refusal_phrases)
        if not is_refusal and len(result["sources"]) == 0:
            logger.error("FAIL: answer given but no sources for '%s'", question[:60])
            passed = False
        else:
            logger.info("PASS: '%s' — %d source(s)", question[:60], len(result["sources"]))
    return passed


def test_empty_query(pipeline):
    """generate_answer() handles empty string without crashing."""
    logger.info("TEST: empty query handling")
    try:
        result = pipeline.generate_answer("")
        assert "answer" in result
        logger.info("PASS: empty query handled gracefully — '%s'", result["answer"][:60])
        return True
    except Exception as e:
        logger.error("FAIL: empty query raised exception — %s", e)
        return False


def test_retrieval_metrics(pipeline):
    """Hit@3 and MRR meet minimum thresholds."""
    logger.info("TEST: retrieval metrics")
    passed = True

    hit_en = hit_at_k(pipeline, questions_en, k=3)
    mrr_en = mean_reciprocal_rank(pipeline, questions_en, k=10)
    hit_ar = hit_at_k(pipeline, questions_ar, k=3)
    mrr_ar = mean_reciprocal_rank(pipeline, questions_ar, k=10)

    logger.info("English — Hit@3: %.2f | MRR: %.3f", hit_en, mrr_en)
    logger.info("Arabic  — Hit@3: %.2f | MRR: %.3f", hit_ar, mrr_ar)

    # Minimum thresholds - derived from notebook results
    thresholds = [
        ("English Hit@3", hit_en, 0.60),
        ("English MRR",   mrr_en, 0.60),
        ("Arabic Hit@3",  hit_ar, 0.30),
        ("Arabic MRR",    mrr_ar, 0.20),
    ]
    for name, value, minimum in thresholds:
        if value < minimum:
            logger.error("FAIL: %s = %.3f — below threshold %.2f", name, value, minimum)
            passed = False
        else:
            logger.info("PASS: %s = %.3f — above threshold %.2f", name, value, minimum)

    return passed

# Test Runner
def run_all_tests():
    logger.info("=" * 60)
    logger.info("Starting test suite — RAGPipeline")
    logger.info("=" * 60)

    # Load pipeline first — required for all other tests
    pipeline = test_pipeline_loads()
    if pipeline is None:
        logger.error("Pipeline failed to load. Aborting all tests.")
        sys.exit(1)

    tests = [
        ("Language detection",          lambda: test_language_detection(pipeline)),
        ("Retrieve returns results",     lambda: test_retrieve_returns_results(pipeline)),
        ("Retrieve language filter",     lambda: test_retrieve_language_filter(pipeline)),
        ("Refusal on unanswerable",      lambda: test_refusal_on_unanswerable(pipeline)),
        ("Answer has sources",           lambda: test_answer_has_sources(pipeline)),
        ("Empty query handling",         lambda: test_empty_query(pipeline)),
        ("Retrieval metrics thresholds", lambda: test_retrieval_metrics(pipeline)),
    ]

    results = []
    for name, test_fn in tests:
        logger.info("-" * 40)
        passed = test_fn()
        results.append((name, passed))

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s", status, name)

    logger.info("-" * 40)
    logger.info("Result: %d/%d passed", passed_count, total)

    if passed_count < total:
        logger.error("Some tests failed.")
        sys.exit(1)
    else:
        logger.info("All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()

