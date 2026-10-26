"""Corpus loading, relevance filtering, and LLM provider clients."""

import json
import pandas as pd
import requests
import time
import re
from pathlib import Path
from typing import Dict, List, Optional


class CorpusLoader:
    """Literature corpus loader for processing various document formats"""

    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.papers = []

    def load_corpus(self, max_papers: int = 500) -> List[Dict]:
        """Load literature corpus files"""
        print("🔍 Scanning literature corpus...")
        self.papers = []

        # Common literature file formats
        paper_files = []
        for ext in ['*.md', '*.json', '*.txt']:
            paper_files.extend(list(self.corpus_path.rglob(ext)))
        paper_files = sorted(set(paper_files), key=lambda path: str(path.relative_to(self.corpus_path)))

        print(f"📚 Found {len(paper_files)} literature files")

        # Load literature content
        for i, file_path in enumerate(paper_files[:max_papers]):
            try:
                paper_content = self.load_single_paper(file_path)
                if paper_content:
                    paper_content['file_path'] = str(file_path)
                    paper_content['paper_id'] = f"paper_{i+1:04d}"
                    self.papers.append(paper_content)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1} papers...")

            except Exception as e:
                print(f"  ⚠️ Error processing file {file_path.name}: {e}")
                continue

        print(f"✅ Successfully loaded {len(self.papers)} papers")
        return self.papers

    def load_single_paper(self, file_path: Path) -> Optional[Dict]:
        """Load a single literature file"""
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return None
                return {
                    'title': data.get('title', ''),
                    'abstract': data.get('abstract', ''),
                    'content': data.get('content', data.get('text', '')),
                    'authors': data.get('authors', []),
                    'year': data.get('year', None),
                    'journal': data.get('journal', ''),
                    'doi': data.get('doi', '')
                }
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Simple txt parsing
                lines = content.split('\n')
                title = lines[0] if lines else ""
                abstract = "\n".join(lines[1:5]) if len(lines) > 1 else ""
                full_content = content

                return {
                    'title': title,
                    'abstract': abstract,
                    'content': full_content,
                    'authors': [],
                    'year': None,
                    'journal': '',
                    'doi': ''
                }
            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Simple markdown parsing
                lines = content.split('\n')
                title = lines[0].replace('#', '').strip() if lines else ""
                abstract = "\n".join(lines[1:5]).strip() if len(lines) > 1 else ""
                full_content = content

                return {
                    'title': title,
                    'abstract': abstract,
                    'content': full_content,
                    'authors': [],
                    'year': None,
                    'journal': '',
                    'doi': ''
                }
            else:
                # Skip other formats for now
                return None

        except Exception as e:
            print(f"  ❌ Failed to read file {file_path.name}: {e}")
            return None


class RelevanceFilter:
    """Literature relevance filter for RuO₂/Er OER papers"""

    def __init__(self):
        # Core keywords - must contain all
        self.core_keywords = [
            r'ruo2|ruthenium dioxide|ruthenium oxide',
            r'oer|oxygen evolution|oxygen evolution reaction',
        ]

        # Related keywords - add relevance weight
        self.related_keywords = [
            # Material related
            r'electrocatalyst|catalyst|electrode',
            r'perovskite|oxide|mixed oxide',
            r'doping|doped|dopant',
            r'rare earth|lanthanide',

            # Performance related
            r'overpotential|over.potential',
            r'ecsa|electrochemical surface area',
            r'durability|stability|lifetime',
            r'tafel slope|tafel',

            # Process related
            r'synthesis|preparation|fabrication',
            r'calcination|annealing|sintering',
            r'temperature|thermal',
            r'loading|mass loading',

            # Mechanism related
            r'oxygen vacancy|vacancy|vacancies',
            r'phase separation|segregation',
            r'interface|interfacial',
            r'mass transport|diffusion|transport'
        ]

    def calculate_relevance_score(self, text: str) -> float:
        """Calculate literature relevance score"""
        if not text:
            return 0.0

        text_lower = text.lower()
        score = 0.0

        # Core keyword matches carry the main relevance weight.
        for keyword_pattern in self.core_keywords:
            if re.search(keyword_pattern, text_lower):
                score += 5.0

        # Related keywords bonus
        for keyword_pattern in self.related_keywords:
            matches = len(re.findall(keyword_pattern, text_lower))
            score += matches * 0.1

        return score

    def filter_relevant_papers(self, papers_df: pd.DataFrame, min_score: float = 3.0) -> pd.DataFrame:
        """Filter relevant literature"""
        if papers_df.empty:
            print("❌ No literature data to filter")
            return pd.DataFrame()

        print("🔍 Calculating literature relevance scores...")

        scores = []
        for _, paper in papers_df.iterrows():
            # Combine title, abstract and content for scoring
            combined_text = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('content', '')[:2000]}"
            score = self.calculate_relevance_score(combined_text)
            scores.append(score)

        scored_papers = papers_df.copy()
        scored_papers['relevance_score'] = scores

        # Filter relevant literature
        relevant_papers = scored_papers[scored_papers['relevance_score'] >= min_score].copy()
        relevant_papers = relevant_papers.sort_values(
            ['relevance_score', 'paper_id'],
            ascending=[False, True],
            kind='mergesort',
        )

        print(f"📊 Filtering results:")
        print(f"  - Input papers: {len(papers_df)}")
        print(f"  - Relevant papers: {len(relevant_papers)}")
        if not relevant_papers.empty:
            print(f"  - Average relevance score: {relevant_papers['relevance_score'].mean():.2f}")
            print(f"  - Highest relevance score: {relevant_papers['relevance_score'].max():.2f}")

        return relevant_papers


class LLMClient:
    """Large Language Model client for various providers"""

    def __init__(self, provider: str = "openai", **kwargs):
        self.provider = provider
        self.config = kwargs

        if provider == "openai":
            # OpenAI API configuration
            self.api_key = kwargs.get("api_key")
            self.model = kwargs.get("model", "gpt-4")
            self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")

        elif provider == "qwen":
            # Qwen API configuration
            self.api_key = kwargs.get("api_key")
            self.model = kwargs.get("model", "qwen-plus")
            self.base_url = kwargs.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

        elif provider == "local":
            # Local model configuration (e.g. ollama)
            self.base_url = kwargs.get("base_url", "http://localhost:11434")
            self.model = kwargs.get("model", "llama2")
            self.api_key = None

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call Large Language Model"""
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._call_openai(prompt)
                elif self.provider == "qwen":
                    return self._call_qwen(prompt)
                elif self.provider == "local":
                    return self._call_local(prompt)
            except Exception as e:
                print(f"  ⚠️ LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

        # This should never be reached due to the raise above, but added for type safety
        raise RuntimeError("Failed to call LLM after all retries")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a materials science expert specializing in electrocatalytic materials. Please extract relevant scientific knowledge and insights based on the provided literature content."},
                {"role": "user", "content": prompt}
            ],
            "temperature": float(self.config.get("temperature", 0.3)),
            "max_tokens": int(self.config.get("max_tokens", 2000)),
        }

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=data,
            timeout=float(self.config.get("timeout", 60)),
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _call_qwen(self, prompt: str) -> str:
        """Call Qwen API using OpenAI-compatible endpoint (works for all Qwen3 models)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a materials science expert specializing in electrocatalytic materials. Please extract relevant scientific knowledge and insights based on the provided literature content."},
                {"role": "user", "content": prompt}
            ],
            "temperature": float(self.config.get("temperature", 0.3)),
            "max_tokens": int(self.config.get("max_tokens", 2000)),
        }

        # Qwen3 thinking models require enable_thinking=False unless you want CoT output
        if "qwen3" in self.model.lower():
            data["enable_thinking"] = False

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=data,
            timeout=float(self.config.get("timeout", 60)),
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _call_local(self, prompt: str) -> str:
        """Call local model (Ollama /chat endpoint for system prompt support)"""
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a materials science expert specializing in electrocatalytic materials. Please extract relevant scientific knowledge and insights based on the provided literature content."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": float(self.config.get("temperature", 0.3)),
                "num_predict": int(self.config.get("max_tokens", 2000)),
            }
        }

        response = requests.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=data,
            timeout=float(self.config.get("timeout", 120)),
        )
        response.raise_for_status()

        return response.json()["message"]["content"]
