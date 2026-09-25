import os

from utils import ServerPreset, ServerProcess


def test_decision_endpoint_batches_constrained_results():
    model_path = os.environ.get("LLAMA_DECISION_TEST_MODEL")
    server = ServerProcess() if model_path else ServerPreset.tinyllama2()
    if model_path:
        server.model_file = model_path
        server.model_hf_repo = None
        server.model_hf_file = None
        server.jinja = True
        server.offline = True
    else:
        server.offline = False
    server.ctk = os.environ.get("LLAMA_DECISION_TEST_CTK")
    server.ctv = os.environ.get("LLAMA_DECISION_TEST_CTV")
    server.n_gpu_layer = 0
    server.n_threads = 4
    server.n_ctx = 512
    server.n_slots = 1
    server.decision_seqs = 8
    try:
        server.start()
        response = server.make_request("POST", "/v1/decision", data={
            "mode": "tree",
            "instructions": "Classify each support request.",
            "schema": {
                "category": {
                    "type": "enum",
                    "choices": ["billing", "technical", "other"],
                    "description": "Request category",
                },
                "urgent": {
                    "type": "boolean",
                    "description": "Needs urgent handling",
                },
            },
            "contexts": ["I was charged twice.", "The application crashes on startup."],
        })

        assert response.status_code == 200, response.body
        assert response.body["object"] == "decision"
        assert len(response.body["results"]) == 2
        for item in response.body["results"]:
            decision = item["decision"]
            assert set(decision) == {"category", "urgent"}
            assert decision["category"] in {"billing", "technical", "other"}
            assert isinstance(decision["urgent"], bool)
            assert item["fields"]["category"]["tree"] is True
            assert item["fields"]["urgent"]["tree"] is True
            assert 0.0 <= item["fields"]["category"]["probability"] <= 1.0
            assert 0.0 <= item["fields"]["urgent"]["probability"] <= 1.0

        invalid = server.make_request("POST", "/v1/decision", data={
            "schema": {"urgent": {"type": "boolean"}},
            "contexts": [],
        })
        assert invalid.status_code == 400
    finally:
        server.stop()
