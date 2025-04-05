from orchestrator import OrchestratorAgent
from context_bank import ContextBank

# Step 1: Mock context bank with minimal info
mock_bank = ContextBank()
document_id = "doc-123"

# Simulate a document being added
mock_bank.documents[document_id] = {
    "id": document_id,
    "metadata": {},
    "content": "Sample NDA Contract Text",
    "report_generated": False
}

# Optionally simulate some progress
mock_bank.clauses[document_id] = [{"id": "clause-1", "text": "Some clause"}]
mock_bank.contradictions[document_id] = [{"id": "c1", "clause_id": "clause-1", "type": "statutory"}]
mock_bank.suggestions[document_id] = {"clause-1": [{"contradiction_id": "c1", "rewritten_text": "..."}]}

# Step 2: Initialize the orchestrator
orch = OrchestratorAgent(use_ollama=False, model_name="test-model", context_bank=mock_bank)

# Step 3: Run the process method
state = {
    "document_id": document_id,
    "task_ledger": []
}

result = orch.process(state)
print("Next step:", result["next_step"])
print("Task ledger:")
for t in result["task_ledger"]:
    print(t)
