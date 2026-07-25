import os
import json
import time
from typing import List, Dict, Any
from groq import Groq

from src.config import config
from src.indexing.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker
from src.generation.generator import GroundedGenerator


class RAGMetricsRunner:
    """Automated LLM-as-a-Judge engine evaluating Faithfulness and Answer Relevancy."""

    def __init__(self, retriever: HybridRetriever, reranker: DocumentReranker, generator: GroundedGenerator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        
        self.judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.judge_model = config.JUDGE_MODEL_NAME

    def _execute_judge_pass(self, prompt: str) -> float:
        """Query LLM judge with structured prompt and extract numerical score."""
        try:
            completion = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            return float(data.get("score", 0.0))
        except Exception as e:
            print(f"[ERROR] Judge evaluation pass failed: {e}")
            return 0.0

    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Measure if the answer is strictly grounded in the retrieved context blocks."""
        joined_context = "\n".join(contexts)
        prompt = (
            f"You are an expert AI Auditor. Rate the FAITHFULNESS of the Answer based strictly on the Context.\n"
            f"Check if every claim in the answer is completely supported by the context. Ignore external knowledge.\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Return JSON format exactly using double quotes: {{\n  \"score\": float\n}} between 0.0 (hallucinated) and 1.0 (perfectly grounded)."
        )
        return self._execute_judge_pass(prompt)

    def calculate_answer_relevancy(self, query: str, answer: str) -> float:
        """Measure how directly the generated answer addresses the user query."""
        prompt = (
            f"You are an expert AI Auditor. Rate the ANSWER RELEVANCY of the generated text to the User Query.\n"
            f"Check if the answer is direct, clear, and addresses the prompt without filler.\n\n"
            f"Query:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Return JSON format exactly using double quotes: {{\n  \"score\": float\n}} between 0.0 (irrelevant) and 1.0 (highly relevant)."
        )
        return self._execute_judge_pass(prompt)

    def run_batch_evaluation(self, test_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute evaluation across the golden test dataset."""
        results = []
        total_faithfulness = 0.0
        total_relevancy = 0.0

        for idx, test_case in enumerate(test_dataset):
            query = test_case["query"]
            print(f"\nEvaluating sample [{idx + 1}/{len(test_dataset)}]: '{query[:60]}...'")
            
            try:
                # 1. Retrieve candidates & rerank
                hybrid_candidates = self.retriever.retrieve(query, top_k=config.RETRIEVAL_TOP_K)
                elite_chunks = self.reranker.rerank(query, hybrid_candidates, top_n=config.RERANK_TOP_N)
                context_strings = [item["chunk"].page_content for item in elite_chunks]

                # 2. Synthesize answer
                llm_output = self.generator.generate_answer(query, elite_chunks)
                generated_answer = llm_output.get("answer", "") if isinstance(llm_output, dict) else str(llm_output)

                # 3. Calculate metrics
                faithfulness = self.calculate_faithfulness(generated_answer, context_strings)
                relevancy = self.calculate_answer_relevancy(query, generated_answer)
                
                print(f" -> Scores | Faithfulness: {faithfulness:.2f} | Relevancy: {relevancy:.2f}")

                results.append({
                    "sample_index": idx + 1,
                    "query": query,
                    "answer": generated_answer,
                    "metrics": {"faithfulness": faithfulness, "answer_relevancy": relevancy}
                })
                
                total_faithfulness += faithfulness
                total_relevancy += relevancy
                
            except Exception as loop_error:
                print(f" -> Warning: Sample {idx + 1} failed: {loop_error}")
            
            # Cooldown to stay within Groq free-tier rate limits
            time.sleep(3)

        total_samples = len(test_dataset) if test_dataset else 1
        summary = {
            "avg_faithfulness": round(total_faithfulness / total_samples, 4),
            "avg_relevancy": round(total_relevancy / total_samples, 4)
        }

        return {"summary": summary, "runs": results}


if __name__ == "__main__":
    print("Initializing components for batch evaluation...")
    from src.indexing.sparse import SparseBM25Index
    from src.indexing.dense import DenseVectorIndex

    sparse = SparseBM25Index()
    dense = DenseVectorIndex()
    sparse.load_index()

    retriever = HybridRetriever(sparse, dense)
    reranker = DocumentReranker()
    generator = GroundedGenerator()

    runner = RAGMetricsRunner(retriever, reranker, generator)

    dataset_path = config.DATA_DIR.parent / "data" / "golden_dataset.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            golden_dataset = json.load(f)
    else:
        golden_dataset = [{"query": "What is the primary objective of the college Code of Conduct?"}]

    print(f"Starting batch evaluation over {len(golden_dataset)} test queries...")
    report = runner.run_batch_evaluation(golden_dataset)
    
    print("\n================ EVALUATION SUMMARY ================")
    print(json.dumps(report["summary"], indent=2))
    print("===================================================\n")