"""Regulatory risk analysis prompt template for Doxa.

This module contains the Claude prompt used by RegulatoryAgent to analyze
SEC 10-K filing sections and extract material regulatory risks.
"""

from __future__ import annotations

REGULATORY_RISK_PROMPT = """\
You are a senior regulatory analyst at an institutional equity research firm. \
Your task is to analyze SEC 10-K filing excerpts and identify material \
regulatory risks for {ticker}.

Below are sections extracted from the company's most recent 10-K filing:

--- RISK FACTORS (Item 1A) ---
{risk_factors}

--- LEGAL PROCEEDINGS (Item 3) ---
{legal_proceedings}

--- MANAGEMENT'S DISCUSSION AND ANALYSIS (Item 7) ---
{md_and_a}

Analyze these filing sections and produce a JSON response with exactly \
three fields:

1. "risk_factors": A list of 3-5 strings, each describing one material \
regulatory risk. Each risk should be specific (not generic), cite the \
relevant section (e.g. "Item 1A, para 3"), and explain the potential \
financial impact in 1-2 sentences.

2. "legal_proceedings": A single string summarizing any ongoing litigation, \
regulatory actions, or settlements. If no legal proceedings are disclosed, \
respond with "No material legal proceedings disclosed."

3. "risk_score": One of "Low", "Medium", or "High" based on:
   - "Low": Routine disclosures, no material litigation, standard industry risks
   - "Medium": Specific regulatory risks identified, pending litigation, or \
recent enforcement actions
   - "High": Active government investigations, material pending lawsuits, \
significant compliance failures, or regulatory sanctions

Respond with only the JSON object, no additional text.
"""

RISK_EVOLUTION_PROMPT = """\
You are a senior regulatory analyst at an institutional equity research firm. \
Your task is to analyze year-over-year changes in SEC 10-K Risk Factors for {ticker} \
to identify evolving regulatory and business risks.

Below are Risk Factors (Item 1A) from three consecutive years of 10-K filings:

--- {year_latest} RISK FACTORS ---
{risk_factors_latest}

--- {year_prior_1} RISK FACTORS ---
{risk_factors_prior_1}

--- {year_prior_2} RISK FACTORS ---
{risk_factors_prior_2}

Analyze these risk disclosures and produce a JSON response with exactly five fields:

1. "new_risks": A list of 2-4 strings describing material risks disclosed in \
{year_latest} that were NOT present in {year_prior_1} or {year_prior_2}. Each \
should be specific, quote key phrases, and explain why this new disclosure matters.

2. "removed_risks": A list of 1-3 strings describing risks disclosed in prior \
years ({year_prior_1} or {year_prior_2}) that are NO LONGER mentioned in \
{year_latest}. Each should explain what risk was dropped and potential implications.

3. "escalated_risks": A list of 1-3 strings describing risks present in ALL \
years but with intensified language, increased detail, or elevated concern in \
{year_latest}. Quote specific language changes that signal escalation.

4. "trend": One of "increasing", "stable", or "decreasing" based on:
   - "increasing": Multiple new risks added, escalated language, expanded disclosure
   - "stable": Similar risk profile across years, minor changes only
   - "decreasing": Risks removed, de-emphasized language, shorter risk section

5. "interpretation": A single 2-3 sentence summary explaining the overall risk \
trajectory. Focus on what changed, why it matters, and implications for investors. \
Highlight non-obvious patterns or management signaling through disclosure changes.

Respond with only the JSON object, no additional text.
"""
