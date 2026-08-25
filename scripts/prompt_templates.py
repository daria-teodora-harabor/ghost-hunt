"""Prompt templates and topics for generating diverse training examples.

Uses multi-axis combinations (Topic x Task Type x Audience/Constraint)
to generate 1,500+ completely distinct prompts without repetition.
"""

from __future__ import annotations

# 60 diverse topics across science, technology, math, humanities, and daily life
TOPICS = [
    # Computer Science & Software
    "binary search trees", "hash collisions", "TCP three-way handshake", "Docker image layers",
    "Kubernetes pod scheduling", "RESTful API design", "JWT token expiry", "database indexing",
    "asynchronous I/O in Python", "microservice circuit breakers", "Git merge vs rebase",
    "garbage collection algorithms", "public key encryption", "SQL injection prevention",
    "caching strategies with Redis", "graph traversal with BFS", "dynamic programming memoization",
    # Physics & Astronomy
    "orbital mechanics", "black hole event horizons", "quantum entanglement", "Doppler effect",
    "thermodynamic entropy", "nuclear fusion in stars", "special relativity time dilation",
    "dark matter evidence", "electromagnetic spectrum", "photoelectric effect",
    # Biology & Chemistry
    "CRISPR-Cas9 gene editing", "cellular respiration", "enzyme catalysis", "DNA replication",
    "mRNA vaccine mechanism", "chemical bonding types", "acid-base titration", "photosynthesis light reactions",
    # Mathematics & Logic
    "Bayes' theorem", "eigenvalues and eigenvectors", "Markov chains", "central limit theorem",
    "Euclidean algorithm", "Fibonacci sequence properties", "prime number distribution",
    # Economics & Finance
    "compound interest dynamics", "supply and demand equilibrium", "inflation causes and effects",
    "game theory Nash equilibrium", "opportunity cost concept", "diversification in portfolio management",
    # Everyday & Practical
    "French press coffee brewing", "indoor houseplant care", "bicycle gear ratios",
    "effective email subject lines", "active listening techniques", "time management time-blocking",
    "sleep hygiene habits", "aerobic vs anaerobic exercise", "fermentation in sourdough bread"
]

# 6 task styles / questions
TASK_STYLES = [
    "Explain the core mechanism of {topic}.",
    "What are the top 3 practical benefits and trade-offs of {topic}?",
    "Provide a beginner-friendly summary of {topic} using a real-world analogy.",
    "Describe how {topic} works step-by-step.",
    "What is the most common misconception about {topic}, and what is the reality?",
    "Write a concise technical guide to {topic}."
]

# 5 audience constraints / formatting instructions
CONSTRAINTS = [
    "Explain in 2 to 3 concise sentences.",
    "Format the explanation with clear bullet points.",
    "Assume the reader is a high school student learning this for the first time.",
    "Focus primarily on real-world practical applications.",
    "Highlight the key underlying principle directly."
]
