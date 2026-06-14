"""Enhanced sentiment analysis prompt for Doxa.

This module contains the Claude prompt used by SentimentAgent to analyze
multi-dimensional market signals including news, insider trading,
short interest, and management contradictions.
"""

from __future__ import annotations

SENTIMENT_ENHANCED_PROMPT = """\
You are a senior hedge fund analyst evaluating market sentiment for {ticker} \
by synthesizing multiple signal streams.

### INPUT DATA

#### 1. RECENT NEWS HEADLINES
{headlines}

#### 2. ALTERNATIVE MARKET SIGNALS
- **Insider Trading:** {insider_trading}
- **Short Interest:** {short_interest}
- **Options Flow:** {options_flow}
- **Social Media Sentiment:** {social_media}

#### 3. MANAGEMENT & REGULATORY CONTEXT (10-K/10-Q)
- **Primary Risk Factors:** {risk_factors}
- **Management Signal:** {management_signal}

### YOUR TASK
Assess the aggregate market sentiment by identifying **CONTRADICTIONS** and \
**CONFLUENCES** between these data points. For example:
- Is management optimistic while insiders are selling?
- Is there positive news but rising short interest?
- Does options flow confirm or contradict the news narrative?

### RESPONSE FORMAT
Respond with a JSON object containing exactly four fields:
- "score": a float between -1.0 (strongly negative / sell pressure) and 1.0 \
(strongly positive / buy pressure)
- "rationale": 2-3 sentences explaining the primary sentiment driver and \
any identified contradictions between data sources
- "contradictions": a list of specific contradictions found (e.g., "Insiders \
selling despite record earnings")
- "key_catalyst": the single most market-moving signal in 10 words or fewer

Respond with only the JSON object, no additional text.
"""
