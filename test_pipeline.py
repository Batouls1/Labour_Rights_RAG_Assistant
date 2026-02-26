
from rag_pipeline import RAGPipeline


evaluation_questions = [
    "What is the legally mandated number of annual leave days?",
    "Are self-employed individuals required to pay municipal fees, and on what basis are they calculated?",
    "Does the guide set minimum pricing standards for freelance services?"
]


def compute_hit_at_k(pipeline, questions, ground_truth, k=3):
    hits = 0
    total = 0

    for question in questions:
        gt = ground_truth[question]
        gt_doc = gt["document"]
        gt_pages = set(str(p) for p in gt["pages"])

        if gt_doc is None:
            continue

        total += 1
        retrieved = pipeline.retrieve(question, k=k)

        for doc in retrieved:
            doc_name = doc["source"].split("\\")[-1]
            page = str(doc["page"])

            if doc_name == gt_doc and page in gt_pages:
                hits += 1
                break

    return hits / total if total > 0 else 0


if __name__ == "__main__":
    pipeline = RAGPipeline()

    ground_truth = {
        "What is the legally mandated number of annual leave days?": {
            "document": "Employment-Rights-Guide.pdf",
            "pages": [16, 24]
        },
        "Are self-employed individuals required to pay municipal fees, and on what basis are they calculated?": {
            "document": "Self-Employment-Guide.pdf",
            "pages": [34]
        },
        "Does the guide set minimum pricing standards for freelance services?": {
            "document": None,
            "pages": []
        }
    }

    hit = compute_hit_at_k(pipeline, evaluation_questions, ground_truth, k=3)

    print(f"Hit@3: {hit:.2f}")

    # Example query
    result = pipeline.generate_answer(
        "Are self-employed individuals required to pay municipal fees?"
    )

    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    print(result["sources"])

