"""
complaint_analysis.py
=====================
Lightweight Student Complaint Categorization and Sentiment/Topic Analysis.
Maps textual student exam feedback into standardized incident categories,
computes temporal distributions, and links feedback to telemetry anomalies.
"""

from typing import Dict, Any, List, Optional
import re
import pandas as pd
import numpy as np

COMPLAINT_CATEGORIES = [
    "Login failure",
    "Question/page loading delay",
    "Answer submission failure",
    "Session disconnection",
    "Generic slowdown",
    "No issue reported"
]

# Rule dictionary mapping keywords and regex expressions to standardized categories
KEYWORD_RULES: Dict[str, List[str]] = {
    "Login failure": [
        r"login", r"log\s*in", r"logged\s*out", r"auth", r"redis", r"session\s*store", r"credentials"
    ],
    "Answer submission failure": [
        r"submit", r"submitting", r"submission", r"saving\s*answer", r"save\s*answer", r"lock\s*timeout", r"500", r"502", r"503"
    ],
    "Question/page loading delay": [
        r"loading\s*next", r"question.*delay", r"formula", r"rendering", r"images", r"stuck\s*at", r"spinning\s*wheel", r"crawling"
    ],
    "Session disconnection": [
        r"disconnect", r"disconnection", r"timed\s*out", r"unreachable", r"dropped", r"reload", r"packet\s*loss", r"ping"
    ],
    "Generic slowdown": [
        r"sluggish", r"slowdown", r"lag", r"timer\s*lag", r"frozen", r"freeze", r"stuttering", r"pegged", r"high\s*server\s*load"
    ],
    "No issue reported": [
        r"normally", r"smoothly", r"fast", r"properly", r"all\s*good", r"expected", r"recovered"
    ]
}


class ComplaintCategorizer:
    """
    Categorizes student complaint texts, aggregates temporal trends,
    and analyzes correlation with infrastructure anomalies.
    """
    def __init__(self):
        self.categories = COMPLAINT_CATEGORIES
        self.compiled_rules = {
            cat: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for cat, patterns in KEYWORD_RULES.items()
        }

    def categorize_text(self, text: str) -> Dict[str, Any]:
        """
        Categorizes a single complaint string based on deterministic keyword regex matching.
        """
        if not text or not isinstance(text, str) or text.strip() == "":
            return {
                "category": "No issue reported",
                "matched_keywords": [],
                "confidence": 0.5
            }

        matched_counts: Dict[str, int] = {cat: 0 for cat in self.categories}
        matched_terms: Dict[str, List[str]] = {cat: [] for cat in self.categories}

        for cat, patterns in self.compiled_rules.items():
            for pat in patterns:
                matches = pat.findall(text)
                if matches:
                    matched_counts[cat] += len(matches)
                    matched_terms[cat].extend(matches)

        # Find category with most hits
        best_cat = "Generic slowdown"
        max_hits = 0

        # Prioritize specific technical issues over generic slowdown
        priority_order = [
            "Answer submission failure",
            "Login failure",
            "Question/page loading delay",
            "Session disconnection",
            "No issue reported",
            "Generic slowdown"
        ]

        for cat in priority_order:
            if matched_counts[cat] > max_hits:
                max_hits = matched_counts[cat]
                best_cat = cat

        confidence = min(0.95, 0.6 + (0.15 * max_hits)) if max_hits > 0 else 0.4

        return {
            "category": best_cat,
            "matched_keywords": list(set(matched_terms[best_cat])),
            "confidence": round(confidence, 2)
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes an entire dataframe containing complaint_text column.
        """
        results = [self.categorize_text(row.get("complaint_text", "")) for _, row in df.iterrows()]
        
        df_out = df.copy()
        df_out["complaint_category"] = [r["category"] for r in results]
        df_out["complaint_match_confidence"] = [r["confidence"] for r in results]
        return df_out

    def get_summary_statistics(self, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes categorical distributions, top examples, and timeline breakdown.
        """
        category_counts = df_processed["complaint_category"].value_counts().to_dict()
        
        # Ensure all categories appear even if zero
        for cat in self.categories:
            if cat not in category_counts:
                category_counts[cat] = 0

        # Collect representative examples for each category
        examples_by_cat = {}
        for cat in self.categories:
            matching = df_processed[df_processed["complaint_category"] == cat]
            if not matching.empty:
                examples_by_cat[cat] = matching["complaint_text"].dropna().unique().tolist()[:3]
            else:
                examples_by_cat[cat] = []

        return {
            "category_distribution": category_counts,
            "representative_examples": examples_by_cat,
            "total_complaints_analyzed": len(df_processed),
            "methodology_note": (
                "Keyword-based deterministic regex categorization. Transparent and deterministic on "
                "small synthetic datasets. Does NOT infer total number of affected students from complaint text."
            )
        }
