from typing import Dict
from pydantic import BaseModel, Field

class InsightCard(BaseModel):
    """Represents a single dashboard card with professional insights."""
    headline: str = Field(
        ..., 
        description="A concise, punchy title summarizing the performance aspect.",
        min_length=5,
        max_length=150
    )
    body: str = Field(
        ..., 
        description="Exactly 2-3 professional sentences comparing performance. Must be grounded in provided metrics.",
        min_length=50,
        max_length=500
    )
    hidden_story: str = Field(
        ..., 
        description="The technical 'why' or trade-off behind the data. Must detail exact comparative performance deltas (e.g., percentage cost advantages or latency speedups).",
        min_length=20,
        max_length=500
    )

class EvaluationReport(BaseModel):
    """The final structured insights report for the UI dashboard."""
    recommendation_summary_text: str = Field(
        ..., 
        description=(
            "A short executive summary less than 400 characters declaring the winning strategy and explaining "
            "why it won based on the balance of accuracy and efficiency. Briefly mention "
            "any critical failures of the losing strategies."
        ),
        min_length=50,
        max_length=400
    )
    winner_rationale: InsightCard = Field(
        ..., 
        description="Deep-dive analysis and metrics explicitly focusing on why the winning strategy succeeded."
    )
    runner_up_rationale: InsightCard = Field(
        ..., 
        description="Deep-dive analysis and metrics explicitly focusing on the runner-up strategy's performance."
    )
    loser_rationale: InsightCard = Field(
        ..., 
        description="Deep-dive analysis explaining the primary bottlenecks or failures of the trailing strategies."
    )
    failure_tags: Dict[str, str] = Field(
        ...,
        description=(
            "A dictionary mapping each evaluated strategy name (key) to a concise 2-4 word diagnosis tag (value) "
            "explaining its primary flaw (e.g., 'Context Cut Off', 'High Redundancy', 'Token Bloat', 'Weak Boundary Health'). "
            "Assign an empty string '' for the overall winning strategy."
        )
    )