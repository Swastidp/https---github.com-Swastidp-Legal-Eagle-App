# utils/legal_processors.py

from __future__ import annotations

import os
import re
import json
from typing import Dict, List, Any

import streamlit as st

from utils.model_cache import get_embedder, get_spacy

# Optional Gemini structured extraction (JSON schema)
try:
    from google import genai
    from google.genai.types import HttpOptions
except Exception:
    genai = None
    HttpOptions = None


class LegalNLPProcessor:
    def __init__(self):
        self._embedder = get_embedder()
        self._nlp = get_spacy()

        # Gemini client for structured entities (optional)
        api_key = (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GEMINI_API_KEY")
        self._gclient = None
        if api_key and genai is not None and HttpOptions is not None:
            try:
                self._gclient = genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))
            except Exception:
                self._gclient = None

        self._gemini_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

        # Feature flags to enable Gemini extraction
        raw_flag = (st.secrets.get("USE_GEMINI_ENTITIES") if hasattr(st, "secrets") else os.getenv("USE_GEMINI_ENTITIES"))
        self._use_gemini_entities = str(raw_flag).lower() in ("1", "true", "yes", "on")

    @property
    def embedder(self):
        return self._embedder

    @property
    def nlp(self):
        return self._nlp

    # ---------------------- Public API ----------------------
    def extract_legal_entities(self, text: str) -> Dict[str, List[Any]]:
        """
        Original 4-bucket entities for your dashboard:
        companies, people, dates, money.
        Uses Gemini structured output if enabled, else rule-based fallback.
        """
        window = text[:20000] if isinstance(text, str) else ""

        # Stage 1: Gemini generic entities (companies, people, dates, money)
        if self._use_gemini_entities and self._gclient:
            g = self._gemini_structured_entities_basic(window)
            if isinstance(g, dict) and any(g.get(k) for k in ["companies", "people", "dates", "money"]):
                return g

        # Stage 2: Rule-based fallback
        return self._extract_legal_entities_rule_based(text)

    def extract_people_companies_with_roles(self, text: str) -> Dict[str, Any]:
        """
        People with designation, organization, relation; companies with party role.
        Returns:
          {
            "people": [{name, designation, organization, relation, mentions[]}],
            "companies": [{name, role_in_document, mentions[]}],
          }
        Uses Gemini structured output first, then a light regex fallback (with a guessed relation).
        """
        result = {"people": [], "companies": []}
        if not text:
            return result

        window = text[:20000]
        if self._gclient and self._use_gemini_entities:
            try:
                r = self._gemini_structured_people_companies(window)
                if isinstance(r, dict) and (r.get("people") or r.get("companies")):
                    return r
            except Exception:
                pass
        return _fallback_people_companies(text)

    def get_legal_embeddings(self, text: str):
        return self.embedder.encode(text)

    def analyze_document_type(self, text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in [
            "whereas", "party agrees", "consideration", "terms and conditions",
            "this agreement", "contract", "covenant", "hereby agree"
        ]):
            return "📄 Contract/Agreement"
        if any(k in t for k in [
            "plaintiff", "defendant", "court", "motion", "your honor",
            "civil action", "case no", "complaint", "docket"
        ]):
            return "⚖️ Legal Filing"
        if any(k in t for k in [
            "board of directors", "shareholders", "bylaws", "articles of incorporation",
            "annual report", "sec filing", "proxy statement"
        ]):
            return "🏢 Corporate Document"
        if any(k in t for k in [
            "policy", "coverage", "premium", "claim", "deductible",
            "insured", "policyholder", "beneficiary"
        ]):
            return "🛡️ Insurance Document"
        if any(k in t for k in [
            "deed", "mortgage", "lease", "rental agreement",
            "property", "landlord", "tenant", "escrow"
        ]):
            return "🏠 Real Estate Document"
        return "📋 General Legal Document"

    # ---------------------- Gemini structured extractors ----------------------
    def _gemini_structured_entities_basic(self, text: str) -> Dict[str, List[str]] | None:
        """
        Strict 4-bucket extraction using response schema, with broadened guidance
        for dates and money to increase recall on global formats.
        """
        if not self._gclient:
            return None
        try:
            config = {
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "companies": {"type": "array", "items": {"type": "string"}},
                        "people": {"type": "array", "items": {"type": "string"}},
                        "dates": {"type": "array", "items": {"type": "string"}},
                        "money": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["companies", "people", "dates", "money"],
                },
            }
            prompt = (
                "Extract entities from the legal text and return only JSON.\n"
                "Instructions:\n"
                "- companies: list registered organization names; prefer those with legal suffixes (Inc., LLC, Ltd., LLP, GmbH, S.A., PLC, Pvt Ltd, etc.). "
                "Include government agencies when clearly acting as parties.\n"
                "- people: human names with at least first+last or honorific+last; exclude headings and all-caps tokens.\n"
                "- dates: capture as they appear, including formats like 01/31/2024, 31-01-2024, 31.01.2024, January 31, 2024, Jan-2024, and month-year mentions such as March 2023.\n"
                "- money: capture amounts with symbols or units in global formats: $, £, €, ₹, ¥, Rs, USD, EUR, GBP, INR, CAD, AUD; "
                "include thousand separators like 1,200,000 or 1.200.000 and Indian grouping like 1,20,000; keep the original surface form.\n"
                'Return strictly: {"companies":[], "people":[], "dates":[], "money":[]}\n\n'
                "TEXT:\n" + (text or "")
            )
            r = self._gclient.models.generate_content(
                model=self._gemini_model,
                contents=prompt,
                config=config,
            )
            parsed = getattr(r, "parsed", None)
            obj = None
            if isinstance(parsed, dict):
                obj = parsed
            else:
                t = (getattr(r, "text", "") or "").strip()
                if t:
                    obj = json.loads(t)
            if not isinstance(obj, dict):
                return None

            def norm(s: str) -> str:
                s = s.strip().strip('"\''"“”‘’()[]{}<>.,;:").strip()
                s = re.sub(r"\s+", " ", s)
                return s

            def uniq(seq: List[str]) -> List[str]:
                seen = set(); out = []
                for s in seq or []:
                    n = norm(str(s))
                    if n and n.lower() not in seen:
                        seen.add(n.lower()); out.append(n)
                return out

            return {
                "companies": uniq(obj.get("companies"))[:10],
                "people": uniq(obj.get("people"))[:10],
                "dates": uniq(obj.get("dates"))[:10],
                "money": uniq(obj.get("money"))[:10],
            }
        except Exception:
            return None

    def _gemini_structured_people_companies(self, text: str) -> Dict[str, Any] | None:
        """
        Returns structured participants:
        { people: [{name, designation, organization, relation, mentions[]}],
          companies: [{name, role_in_document, mentions[]}] }
        """
        if not self._gclient:
            return None
        try:
            schema = {
                "type": "object",
                "properties": {
                    "people": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "designation": {"type": "string"},
                                "organization": {"type": "string"},
                                "relation": {"type": "string"},
                                "mentions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["name"],
                        },
                    },
                    "companies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "role_in_document": {"type": "string"},
                                "mentions": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["name"],
                        },
                    },
                },
                "required": ["people", "companies"],
            }
            prompt = (
                "From the following legal text, extract participants with structure.\n"
                "Rules:\n"
                "- people: include a designation/title when present (e.g., CEO, Director, Counsel), "
                "  the organization if explicitly linked (e.g., 'John Smith, CTO of Acme LLC'), "
                "  and one short relation sentence explaining who they are or how they relate to this document "
                "  (e.g., 'Lead counsel for Plaintiff', 'CTO of Acme LLC', 'Notary witnessing execution').\n"
                "- companies: include the party role in the document when present (e.g., Plaintiff, Defendant, "
                "  Supplier, Customer, Lender, Borrower, Licensor, Licensee, Seller, Buyer).\n"
                "- For each entity, add 1–2 short quotes (mentions) from the text that justify the extraction.\n"
                "- Return only JSON that conforms exactly to the provided schema.\n\n"
                "TEXT:\n" + (text or "")
            )
            r = self._gclient.models.generate_content(
                model=self._gemini_model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            parsed = getattr(r, "parsed", None)
            if isinstance(parsed, dict):
                return _dedupe_roles(parsed)
            t = (getattr(r, "text", "") or "").strip()
            if t:
                return _dedupe_roles(json.loads(t))
            return None
        except Exception:
            return None

    # ---------------------- Rule-based fallback (used by extract_legal_entities) ----------------------
    def _extract_legal_entities_rule_based(self, text: str) -> Dict[str, List[Any]]:
        def norm(s: str) -> str:
            s = s.strip().strip('"\''"“”‘’()[]{}<>.,;:").strip()
            s = re.sub(r"\s+", " ", s)
            return s

        def uniq_keep_first(seq: List[str]) -> List[str]:
            seen_lower = set()
            out = []
            for s in seq:
                n = norm(s)
                if not n:
                    continue
                k = n.lower()
                if k not in seen_lower:
                    seen_lower.add(k)
                    out.append(n)
            return out

        STOP_WORDS = set([
            "agreement", "contract", "company", "party", "parties", "board",
            "section", "article", "clause", "schedule", "exhibit", "annex",
            "hereinafter", "witnesseth", "whereas",
        ])
        MONTHS = set([
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ])

        companies_spacy: List[str] = []
        people_spacy: List[str] = []
        spacy_dates: List[str] = []
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    et = ent.label_
                    val = norm(ent.text)
                    if not val:
                        continue
                    low_tokens = val.lower().split()
                    if any(w in STOP_WORDS for w in low_tokens):
                        continue
                    if et == "ORG":
                        if len(val) >= 3:
                            companies_spacy.append(val)
                    elif et == "PERSON":
                        if len(val.split()) >= 2 or re.match(r"^(Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Hon\.|Prof\.|Justice)\s+[A-Z]", val):
                            if not re.fullmatch(r"[A-Z ]{3,}", val) and not any(tok in MONTHS for tok in low_tokens):
                                people_spacy.append(val)
                    elif et == "DATE":
                        spacy_dates.append(val)
            except Exception as e:
                st.warning(f"spaCy processing failed: {e}")

        # Tighter company/person regex
        company_suffix = r"(?:Inc\.?|Incorporated|LLC|L\.L\.C\.|Ltd\.?|Limited|LLP|L\.L\.P\.|PLC|P\.L\.C\.|Pvt\.?\s+Ltd\.?|GmbH|S\.A\.|S\.A\.S\.?|N\.V\.|B\.V\.)"
        company_pat = re.compile(
            rf"\b([A-Z][A-Za-z&,\-.' ]+?\s*,?\s*{company_suffix})\b"
        )
        honorific = r"(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Hon\.|Prof\.|Justice)"
        person_pat = re.compile(
            rf"\b(?:{honorific}\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+(?:Jr\.?|Sr\.?|III|IV))?)\b"
        )

        companies_rx: List[str] = [m.group(1) for m in company_pat.finditer(text)]
        people_rx: List[str] = [m.group(1) for m in person_pat.finditer(text)]

        def filter_company(name: str) -> bool:
            n = name.strip()
            if len(n) < 3:
                return False
            if n.lower() in STOP_WORDS:
                return False
            return True

        def filter_person(name: str) -> bool:
            n = name.strip()
            if len(n.split()) < 2:
                return False
            low_tokens = n.lower().split()
            if any(tok in MONTHS for tok in low_tokens):
                return False
            if re.fullmatch(r"[A-Z ]{3,}", n):
                return False
            return True

        companies_all = [c for c in (companies_spacy + companies_rx) if filter_company(c)]
        people_all = [p for p in (people_spacy + people_rx) if filter_person(p)]

        # Broadened money and dates patterns (global formats)
        money_patterns = [
            r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?",
            r"[£€¥₹]\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?",
            r"Rs\.?\s?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?",
            r"(?:USD|EUR|GBP|INR|CAD|AUD)\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?",
            r"\d{1,3}(?:,\d{3})+(?:\.\d+)?\s?(?:USD|EUR|GBP|INR|dollars?|euros?|pounds?|rupees?)",
        ]
        money: List[str] = []
        for pat in money_patterns:
            money.extend(re.findall(pat, text, flags=re.IGNORECASE))

        date_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
            r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
            r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4}\b",
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
            r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b",
        ]
        regex_dates: List[str] = []
        for pat in date_patterns:
            regex_dates.extend(re.findall(pat, text, flags=re.IGNORECASE))

        return {
            "dates": uniq_keep_first(spacy_dates + regex_dates)[:10],
            "money": uniq_keep_first(money)[:10],
            "companies": uniq_keep_first(companies_all)[:10],
            "people": uniq_keep_first(people_all)[:10],
        }


# ---------------------- Helpers for structured role extraction ----------------------
def _dedupe_roles(obj: Dict[str, Any]) -> Dict[str, Any]:
    def norm(s: str) -> str:
        s = s.strip().strip('"\''"“”‘’()[]{}<>.,;:").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    out = {"people": [], "companies": []}
    seen_p, seen_c = set(), set()

    for p in (obj.get("people") or []):
        name = norm(str(p.get("name", "")))
        if not name:
            continue
        key = name.lower()
        if key in seen_p:
            continue
        seen_p.add(key)
        out["people"].append({
            "name": name,
            "designation": norm(str(p.get("designation", ""))) if p.get("designation") else "",
            "organization": norm(str(p.get("organization", ""))) if p.get("organization") else "",
            "relation": norm(str(p.get("relation", ""))) if p.get("relation") else "",
            "mentions": [norm(m) for m in (p.get("mentions") or [])][:2],
        })

    for c in (obj.get("companies") or []):
        name = norm(str(c.get("name", "")))
        if not name:
            continue
        key = name.lower()
        if key in seen_c:
            continue
        seen_c.add(key)
        out["companies"].append({
            "name": name,
            "role_in_document": norm(str(c.get("role_in_document", ""))) if c.get("role_in_document") else "",
            "mentions": [norm(m) for m in (c.get("mentions") or [])][:2],
        })

    return out


def _fallback_people_companies(text: str) -> Dict[str, Any]:
    # Lightweight regex fallback with a guessed relation based on local context
    people = []
    companies = []

    honorific = r"(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Hon\.|Prof\.|Justice)"
    person_pat = re.compile(rf"\b(?:{honorific}\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

    def _guess_relation(name: str, context: str) -> str:
        patt = re.compile(rf"{re.escape(name)}[^.\n]{{0,160}}", re.IGNORECASE)
        m = patt.search(context)
        span = m.group(0) if m else ""
        desig = ""
        for kw in ["CEO","Chief Executive Officer","CFO","CTO","Director","Counsel","Attorney","Partner","Manager","Notary","Authorized Signatory","Witness"]:
            if re.search(rf"\b{kw}\b", span, re.IGNORECASE):
                desig = kw; break
        org = ""
        m2 = re.search(r"\bof\s+([A-Z][A-Za-z0-9&.,\- ]+?(?:Inc\.?|LLC|Ltd\.?|LLP|GmbH|S\.A\.|PLC)?)\b", span)
        if m2: org = m2.group(1)
        if desig and org: return f"{desig} at {org}"
        if desig: return desig
        if org: return f"Associated with {org}"
        return ""

    head = text[:4000]
    for m in person_pat.finditer(text[:10000]):
        pname = m.group(1)
        people.append({"name": pname, "designation": "", "organization": "", "relation": _guess_relation(pname, head), "mentions": []})

    company_suffix = r"(?:Inc\.?|Incorporated|LLC|Ltd\.?|Limited|LLP|PLC|GmbH|S\.A\.|Pvt\.?\s+Ltd\.?)"
    comp_pat = re.compile(rf"\b([A-Z][A-Za-z&,\-.' ]+?\s*,?\s*{company_suffix})\b")
    for m in comp_pat.finditer(text[:10000]):
        companies.append({"name": m.group(1), "role_in_document": "", "mentions": []})

    return {"people": people[:8], "companies": companies[:8]}
