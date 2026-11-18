"""
jobshield_model.py
==================
Core Machine Learning, NLP Vectorization, Feature Engineering, and Explainability Engine
for JobShield AI.

Architecture:
1. JobShieldTfidfVectorizer:
   - Normalizes emails, free email providers (gmail/yahoo/hotmail/etc.), phone numbers,
     URLs, and currency symbols into canonical tokens that bridge EMSCAD dataset tokens
     (#EMAIL_...#, #PHONE_...#, #URL_...#) with live real-world documents.
   - Extracts sublinear TF-IDF unigrams and bigrams (up to 75,000 features) with L2 normalization.

2. JobShieldHybridModel (4-Branch Calibrated Ensemble):
   - Branch 1 (Structured + N-Gram NBSVM Logistic Regression): Combines Naive Bayes
     log-count ratio (r) weighted n-grams with 11 scaled structural & linguistic features.
   - Branch 2 (Pure-Text N-Gram NBSVM Logistic Regression): Trained strictly on text n-grams
     so live document uploads without Kaggle CSV flags remain un-biased.
   - Branch 3 (Sparse TF-IDF Cosine k-NN Memory Bank): Computes similarity against indexed
     training postings to detect templated scam campaigns and retrieve similar cases.
   - Branch 4 (Domain Red-Flag & Legitimacy Risk Calibrator): Evaluates 16 domain-specific
     linguistic, financial-scam, contact-anomaly, vagueness, and corporate-legitimacy signals.
"""

import re
import json
import pickle
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Domain Lexicons & Pattern Definitions
# ---------------------------------------------------------------------------

URGENCY_PHRASES = [
    "apply now", "urgent", "immediate start", "limited time", "act fast",
    "hurry", "don't miss", "do not miss", "immediate hire", "start immediately",
    "hiring immediately", "instant hiring", "spots left", "limited spots",
    "apply today", "urgent requirement", "immediate joiner", "join immediately",
    "last chance", "closing soon", "apply immediately", "urgent hiring",
    "within 24 hours", "same day hire", "instant offer"
]

HIGH_RISK_SCAM_PHRASES = [
    "registration fee", "training fee", "security deposit", "processing fee",
    "application fee", "equipment fee", "background check fee", " refundable fee",
    "wire transfer", "western union", "moneygram", "bitcoin", "crypto", "usdt",
    "cashier's check", "cashier check", "money order", "bank account details",
    "routing number", "social security number", "ssn", "credit card details",
    "mystery shopper", "envelope stuffing", "reshipping", "package forwarding",
    "payment processor", "check cashing", "google chat interview", "telegram",
    "whatsapp", "signal app", "text only interview", "no interview required",
    "no interview", "guaranteed income", "be your own boss", "financial freedom",
    "unlimited earning", "get rich", "easy money", "earn passive income"
]

MEDIUM_RISK_SCAM_PHRASES = [
    "work from home", "work at home", "data entry", "no experience",
    "no experience needed", "no experience required", "no degree required",
    "earn up to", "per week", "per day", "weekly pay", "daily pay",
    "daily payout", "weekly payout", "instant payout", "cash payout",
    "typing job", "copy paste", "form filling", "ad posting",
    "part time home", "home based", "clerk", "accountant needed urgently"
]

LEGITIMACY_PHRASES = [
    "equal opportunity employer", "401(k)", "401k", "health insurance",
    "dental", "vision", "paid time off", "pto", "parental leave",
    "tuition reimbursement", "bachelor's degree", "bachelor degree",
    "master's degree", "years of experience", "cross-functional",
    "stakeholders", "agile", "scrum", "code review", "ci/cd",
    "architecture", "compliance", "affirmative action", "veteran status",
    "disability", "reasonable accommodation", "competitive salary",
    "comprehensive benefits", "professional development", "mentorship",
    "performance bonus", "equity", "stock options", "commuter benefits"
]

COMPANY_SECTION_MARKERS = [
    "about us", "company profile", "who we are", "our company", "founded in",
    "headquartered", "our mission", "about the company", "company overview",
    "leading provider", "global leader", "we are a"
]

REQUIREMENTS_MARKERS = [
    "requirements", "qualifications", "what you bring", "skills required",
    "minimum qualifications", "preferred qualifications", "experience in",
    "proficient in", "knowledge of", "bachelor", "degree", "certification"
]

BENEFITS_MARKERS = [
    "benefits", "perks", "what we offer", "compensation", "medical",
    "dental", "401k", "401(k)", "vacation", "paid leave", "insurance",
    "wellness", "flexible hours", " provident fund", "gratuity"
]

EDUCATION_MARKERS = [
    "bachelor", "master", "phd", "degree", "diploma", "b.tech", "b.e.",
    "b.sc", "m.tech", "mba", "computer science", "high school", "ged",
    "associate's", "university", "college", "graduate"
]

FREE_EMAIL_RE = re.compile(
    r'[\w\.-]+@(?:gmail|yahoo|hotmail|outlook|aol|icloud|ymail|rediffmail|protonmail|mail|zoho|gmx)\.[a-z]{2,}',
    re.IGNORECASE
)
ANY_EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.[a-z]{2,}', re.IGNORECASE)
PHONE_RE = re.compile(r'(?:\+?\d{1,3}[\s-]?)?\(?\d{3,5}\)?[\s.-]?\d{3,5}[\s.-]?\d{3,5}')
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
SALARY_BAIT_RE = re.compile(
    r'(?:\$|₹|rs\.?\s*|usd\s*|inr\s*|£|€)\s*\d[\d,]*(?:\s*-\s*(?:\$|₹|rs\.?\s*)?\d[\d,]*)?\s*(?:/|\s+per\s+|\s+a\s+)(?:day|week|hr|hour|wk)',
    re.IGNORECASE
)
TOKEN_RE = re.compile(r'[a-z0-9_]{2,}')


def normalize_job_text(text: str) -> str:
    """Normalizes raw job text and bridges EMSCAD anonymized tokens with live text."""
    if not text:
        return ""
    t = str(text)
    # Preserve raw markers before lowercasing
    t_lower = t.lower()
    t_lower = re.sub(r'#email_[a-f0-9]+#', ' __email_token__ ', t_lower)
    t_lower = re.sub(r'#phone_[a-f0-9]+#', ' __phone_token__ ', t_lower)
    t_lower = re.sub(r'#url_[a-f0-9]+#', ' __url_token__ ', t_lower)
    t_lower = FREE_EMAIL_RE.sub(' __email_token__ __free_email_token__ ', t_lower)
    t_lower = ANY_EMAIL_RE.sub(' __email_token__ ', t_lower)
    t_lower = URL_RE.sub(' __url_token__ ', t_lower)
    t_lower = PHONE_RE.sub(' __phone_token__ ', t_lower)
    return t_lower


def extract_ngrams_from_normalized(norm_text: str):
    tokens = TOKEN_RE.findall(norm_text)
    if not tokens:
        return []
    bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
    return tokens + bigrams


def count_urgency(text: str) -> int:
    t = str(text).lower()
    return sum(t.count(w) for w in URGENCY_PHRASES)


def hype_score(text: str) -> int:
    s = str(text)
    exclamations = s.count("!")
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", s))
    multi_dollar = len(re.findall(r"\${2,}", s))
    return exclamations + caps_words + (multi_dollar * 2)


def extract_detailed_signals(
    text: str,
    company_profile: str = None,
    required_education: str = None,
    industry: str = None,
    benefits: str = None,
    telecommuting: int = None,
    has_company_logo: int = None,
    has_questions: int = None,
) -> dict:
    """
    Extracts comprehensive linguistic, structural, red-flag, and legitimacy signals
    from either raw unstructured text (PDF/Image/DOCX/TXT/pasted) or structured rows.
    """
    raw_text = str(text or "")
    lower_text = raw_text.lower()
    norm_text = normalize_job_text(raw_text)
    words = raw_text.split()
    word_count = len(words)

    # 1. Urgency phrases matched
    matched_urgency = [w for w in URGENCY_PHRASES if w in lower_text]
    urgency_cnt = sum(lower_text.count(w) for w in URGENCY_PHRASES)

    # 2. High-risk & medium-risk scam phrases matched
    matched_high_scam = [w for w in HIGH_RISK_SCAM_PHRASES if w in lower_text]
    matched_med_scam = [w for w in MEDIUM_RISK_SCAM_PHRASES if w in lower_text]
    high_scam_cnt = sum(lower_text.count(w) for w in HIGH_RISK_SCAM_PHRASES)
    med_scam_cnt = sum(lower_text.count(w) for w in MEDIUM_RISK_SCAM_PHRASES)

    # 3. Contact anomalies (free emails, direct phone/email lures in body)
    free_emails = FREE_EMAIL_RE.findall(raw_text)
    all_emails = ANY_EMAIL_RE.findall(raw_text)
    phones = PHONE_RE.findall(raw_text)
    has_emscad_email = "#email_" in lower_text or "__email_token__" in norm_text
    has_emscad_phone = "#phone_" in lower_text or "__phone_token__" in norm_text

    # 4. Salary bait patterns
    salary_baits = SALARY_BAIT_RE.findall(raw_text)

    # 5. Legitimacy corporate phrases matched
    matched_legit = [w for w in LEGITIMACY_PHRASES if w in lower_text]
    legit_cnt = sum(lower_text.count(w) for w in LEGITIMACY_PHRASES)

    # 6. Vagueness assessment (works both on structured columns and unstructured documents)
    has_comp = (
        (company_profile is not None and str(company_profile).strip() != "")
        or any(m in lower_text for m in COMPANY_SECTION_MARKERS)
    )
    has_edu = (
        (required_education is not None and str(required_education).strip() != "")
        or any(m in lower_text for m in EDUCATION_MARKERS)
    )
    has_ind_or_req = (
        (industry is not None and str(industry).strip() != "")
        or any(m in lower_text for m in REQUIREMENTS_MARKERS)
    )
    has_ben = (
        (benefits is not None and str(benefits).strip() != "")
        or any(m in lower_text for m in BENEFITS_MARKERS)
    )

    missing_sections = []
    if not has_comp:
        missing_sections.append("Company Profile / Background")
    if not has_edu:
        missing_sections.append("Education / Degree Qualifications")
    if not has_ind_or_req:
        missing_sections.append("Detailed Role Requirements / Industry")
    if not has_ben:
        missing_sections.append("Benefits & Compensation Structure")

    vagueness = len(missing_sections)

    # 7. Hype & formatting metrics
    exclamations = raw_text.count("!")
    caps_words = re.findall(r"\b[A-Z]{3,}\b", raw_text)
    hype = exclamations + len(caps_words)

    # 8. Inferred metadata flags when not explicitly provided
    if telecommuting is None:
        telecommuting = int(any(w in lower_text for w in ["work from home", "work at home", "remote", "telecommute", "home based"]))
    if has_company_logo is None:
        has_company_logo = int(has_comp and legit_cnt >= 2 and len(free_emails) == 0 and high_scam_cnt == 0)
    if has_questions is None:
        has_questions = int(has_ind_or_req and word_count >= 120 and high_scam_cnt == 0)

    scam_Composite = (
        high_scam_cnt * 2
        + med_scam_cnt
        + int(has_emscad_email)
        + int(has_emscad_phone)
        + len(free_emails) * 2
        + len(salary_baits)
    )

    return {
        "urgency_score": int(urgency_cnt),
        "matched_urgency": matched_urgency,
        "high_scam_count": int(high_scam_cnt),
        "matched_high_scam": matched_high_scam,
        "med_scam_count": int(med_scam_cnt),
        "matched_med_scam": matched_med_scam,
        "scam_composite_score": int(scam_Composite),
        "legitimacy_score": int(legit_cnt),
        "matched_legitimacy": matched_legit,
        "vagueness_score": int(vagueness),
        "missing_sections": missing_sections,
        "text_length": int(word_count),
        "char_length": len(raw_text),
        "hype_score": int(hype),
        "exclamation_count": int(exclamations),
        "caps_word_count": len(caps_words),
        "free_emails": free_emails,
        "all_emails": all_emails,
        "phones": phones,
        "salary_baits": salary_baits,
        "telecommuting": int(telecommuting),
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
    }


class JobShieldTfidfVectorizer:
    """
    Pickle-safe, version-independent Sublinear TF-IDF Unigram + Bigram Vectorizer.
    """

    def __init__(self, max_features: int = 75000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab_ = {}
        self.vocab_items_ = []
        self.idf_ = None

    def fit(self, raw_texts):
        df_counts = Counter()
        n_docs = len(raw_texts)
        for txt in raw_texts:
            norm = normalize_job_text(txt)
            df_counts.update(set(extract_ngrams_from_normalized(norm)))

        vocab_items = [
            tok for tok, cnt in df_counts.most_common() if cnt >= self.min_df
        ][: self.max_features]
        self.vocab_items_ = vocab_items
        self.vocab_ = {tok: i for i, tok in enumerate(vocab_items)}
        self.idf_ = np.array(
            [
                np.log((1.0 + n_docs) / (1.0 + df_counts[tok])) + 1.0
                for tok in vocab_items
            ],
            dtype=np.float32,
        )
        return self

    def transform(self, raw_texts):
        rows, cols, vals = [], [], []
        vocab = self.vocab_
        idf = self.idf_
        for r, txt in enumerate(raw_texts):
            norm = normalize_job_text(txt)
            tf = Counter(
                tok for tok in extract_ngrams_from_normalized(norm) if tok in vocab
            )
            if not tf:
                continue
            for tok, cnt in tf.items():
                c = vocab[tok]
                rows.append(r)
                cols.append(c)
                vals.append((1.0 + np.log(cnt)) * idf[c])

        mat = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(raw_texts), len(vocab)),
            dtype=np.float32,
        )
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        mat = (sparse.diags(1.0 / norms, dtype=np.float32) @ mat).tocsr()
        return mat

    def fit_transform(self, raw_texts):
        self.fit(raw_texts)
        return self.transform(raw_texts)


class JobShieldHybridModel:
    """
    4-Branch Calibrated Ensemble for Fake Job Detection:
      1. Structured + Text NBSVM L2 Logistic Regression
      2. Pure-Text NBSVM L2 Logistic Regression
      3. Sparse TF-IDF Cosine k-NN Memory Bank
      4. Expert Heuristic & Red-Flag Calibrator
    Compatible with both standard sklearn-style `.predict(X)` / `.predict_proba(X)`
    AND rich document analysis `.analyze_posting(...)`.
    """

    def __init__(self, threshold: float = 0.40, knn_weight: float = 0.25):
        self.threshold = threshold
        self.knn_weight = knn_weight
        self.r_ = None
        self.w_struct_ = None
        self.b_struct_ = 0.0
        self.w_text_ = None
        self.b_text_ = 0.0
        self.f_mean_ = None
        self.f_std_ = None
        self.X_train_tfidf_ = None
        self.y_train_ = None
        self.train_titles_ = None
        self.vocab_items_ = None
        self.metrics_summary_ = {}

    @staticmethod
    def _train_logreg_lbfgs(X, y, pos_weight=4.5, C=10.0, maxiter=200):
        X64 = X.astype(np.float64)
        y64 = y.astype(np.float64)
        n_features = X64.shape[1]
        sample_weights = np.where(y64 == 1, pos_weight, 1.0)

        def loss_and_grad(w):
            wb, b = w[:-1], w[-1]
            z = X64 @ wb + b
            prob = np.where(
                z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z))
            )
            pos_loss = np.where(
                z >= 0, np.log1p(np.exp(-z)), -z + np.log1p(np.exp(z))
            )
            neg_loss = np.where(
                z >= 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z))
            )
            loss = np.sum(
                sample_weights * (y64 * pos_loss + (1.0 - y64) * neg_loss)
            ) + 0.5 * (1.0 / C) * np.dot(wb, wb)
            diff = sample_weights * (prob - y64)
            grad_w = (X64.T @ diff) + (1.0 / C) * wb
            grad_b = np.sum(diff)
            return loss, np.append(grad_w, grad_b)

        res = minimize(
            loss_and_grad,
            np.zeros(n_features + 1, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter},
        )
        return res.x[:-1].astype(np.float32), float(res.x[-1])

    def fit(self, X_tfidf, numeric_features, y, train_titles=None, vocab_items=None):
        y = np.asarray(y, dtype=np.int32)
        pos_mask = y == 1
        neg_mask = y == 0
        alpha = 0.35
        p = np.asarray(X_tfidf[pos_mask].sum(axis=0)).ravel().astype(np.float64) + alpha
        q = np.asarray(X_tfidf[neg_mask].sum(axis=0)).ravel().astype(np.float64) + alpha
        self.r_ = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)

        X_nb = (X_tfidf @ sparse.diags(self.r_, dtype=np.float32)).tocsr()

        F = np.asarray(numeric_features, dtype=np.float32)
        self.f_mean_ = F.mean(axis=0)
        self.f_std_ = F.std(axis=0) + 1e-6
        F_scaled = ((F - self.f_mean_) / self.f_std_) * 0.25

        X_all = sparse.hstack([X_nb, F_scaled], format="csr", dtype=np.float32)

        self.w_struct_, self.b_struct_ = self._train_logreg_lbfgs(
            X_all, y, pos_weight=4.5, C=10.0, maxiter=200
        )
        self.w_text_, self.b_text_ = self._train_logreg_lbfgs(
            X_nb, y, pos_weight=4.5, C=10.0, maxiter=200
        )

        # Prune TF-IDF matrix to keep top 35 weights per row for compact, fast k-NN
        self.X_train_tfidf_ = self._compress_csr(X_tfidf, top_n=150)
        self.y_train_ = y
        self.train_titles_ = list(train_titles) if train_titles is not None else []
        self.vocab_items_ = list(vocab_items) if vocab_items is not None else []
        return self

    @staticmethod
    def _compress_csr(mat: sparse.csr_matrix, top_n: int = 150) -> sparse.csr_matrix:
        mat = mat.tocsr()
        indptr = [0]
        indices = []
        data = []
        for i in range(mat.shape[0]):
            start, end = mat.indptr[i], mat.indptr[i + 1]
            row_idx = mat.indices[start:end]
            row_val = mat.data[start:end]
            if len(row_val) > top_n:
                top_pos = np.argpartition(row_val, -top_n)[-top_n:]
                row_idx = row_idx[top_pos]
                row_val = row_val[top_pos]
                norm = np.sqrt(np.sum(row_val ** 2))
                if norm > 0:
                    row_val = row_val / norm
            indices.extend(row_idx)
            data.extend(row_val)
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
            shape=mat.shape,
        )

    def _knn_predict_proba(self, X_tfidf):
        S = (X_tfidf @ self.X_train_tfidf_.T).toarray()
        top_k = 7
        idx_part = np.argpartition(S, -top_k, axis=1)[:, -top_k:]
        rows_grid = np.arange(S.shape[0])[:, None]
        top_sims = S[rows_grid, idx_part]
        top_labels = self.y_train_[idx_part]
        weights = np.maximum(top_sims, 0.0) ** 4 + 1e-6
        probs_knn = np.sum(weights * top_labels, axis=1) / np.sum(weights, axis=1)
        max_sim = np.max(top_sims, axis=1)
        return probs_knn, max_sim, idx_part, top_sims

    def predict_proba_branches(self, X_tfidf, numeric_features=None, use_structured=True):
        X_nb = (X_tfidf @ sparse.diags(self.r_, dtype=np.float32)).tocsr()
        logits_text = X_nb @ self.w_text_ + self.b_text_
        probs_text = 1.0 / (1.0 + np.exp(-np.clip(logits_text, -30, 30)))

        if numeric_features is not None and use_structured:
            F = np.asarray(numeric_features, dtype=np.float32)
            # Support legacy 7-feature input from old member4_app.py or full 11-feature input
            if F.shape[1] == 7:
                # Expand 7 features [urg, vag, tlen, hype, tele, logo, ques] to 11 features
                urg = F[:, 0]
                vag = F[:, 1]
                tlen = np.log1p(np.maximum(F[:, 2], 0))
                scam = np.zeros_like(urg)
                clen = np.where(vag == 0, 5.5, 2.0)
                rlen = np.where(vag == 0, 5.5, 3.0)
                blen = np.where(vag == 0, 5.0, 2.0)
                has_sal = np.zeros_like(urg)
                tele = F[:, 4]
                logo = F[:, 5]
                ques = F[:, 6]
                F = np.column_stack([urg, scam, vag, tlen, clen, rlen, blen, has_sal, tele, logo, ques])
            F_scaled = ((F - self.f_mean_) / self.f_std_) * 0.25
            X_all = sparse.hstack([X_nb, F_scaled], format="csr", dtype=np.float32)
            logits_struct = X_all @ self.w_struct_ + self.b_struct_
            probs_struct = 1.0 / (1.0 + np.exp(-np.clip(logits_struct, -30, 30)))
        else:
            probs_struct = probs_text

        probs_knn, max_sim, idx_part, top_sims = self._knn_predict_proba(X_tfidf)

        if use_structured and numeric_features is not None:
            base_prob = 0.75 * probs_struct + 0.25 * probs_text
            final_prob = (1.0 - self.knn_weight) * base_prob + self.knn_weight * probs_knn
        else:
            # For unstructured live text, rely primarily on pure-text NBSVM + kNN when similarity is strong
            dyn_knn_w = np.where(max_sim >= 0.35, 0.30, 0.12)
            base_prob = 0.70 * probs_text + 0.30 * probs_struct
            final_prob = (1.0 - dyn_knn_w) * base_prob + dyn_knn_w * probs_knn

        return {
            "final_prob": final_prob,
            "struct_prob": probs_struct,
            "text_prob": probs_text,
            "knn_prob": probs_knn,
            "max_sim": max_sim,
            "knn_indices": idx_part,
            "knn_sims": top_sims,
        }

    def predict_proba(self, X_input):
        """
        Standard scikit-learn compatible predict_proba(X).
        Accepts either X_tfidf alone or hstack([X_tfidf, numeric_features]).
        """
        n_vocab = len(self.r_)
        if X_input.shape[1] > n_vocab:
            X_csr = X_input.tocsr()
            X_tfidf = X_csr[:, :n_vocab]
            F_num = X_csr[:, n_vocab:].toarray()
            res = self.predict_proba_branches(X_tfidf, F_num, use_structured=True)
        else:
            res = self.predict_proba_branches(X_input, None, use_structured=False)
        p1 = res["final_prob"]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X_input):
        probs = self.predict_proba(X_input)[:, 1]
        return (probs >= self.threshold).astype(int)

    def analyze_posting(
        self,
        vectorizer: JobShieldTfidfVectorizer,
        text: str,
        title: str = "",
        company_profile: str = None,
        requirements: str = None,
        benefits: str = None,
        salary_range: str = None,
        required_education: str = None,
        industry: str = None,
        telecommuting: int = None,
        has_company_logo: int = None,
        has_questions: int = None,
        use_explicit_metadata: bool = False,
    ) -> dict:
        """
        Comprehensive end-to-end analysis for live text or extracted document content.
        Combines the 4-branch ensemble and generates token-level & rule-level explanations.
        """
        full_raw = " ".join(
            filter(
                None,
                [
                    title,
                    title,
                    company_profile,
                    text,
                    requirements,
                    benefits,
                    salary_range,
                    required_education,
                    industry,
                ],
            )
        ).strip()

        signals = extract_detailed_signals(
            full_raw,
            company_profile=company_profile,
            required_education=required_education,
            industry=industry,
            benefits=benefits,
            telecommuting=telecommuting,
            has_company_logo=has_company_logo,
            has_questions=has_questions,
        )

        X_tfidf = vectorizer.transform([full_raw])
        norm_text = normalize_job_text(full_raw)

        # Build 11-element numeric vector
        urg = float(signals["urgency_score"])
        scam = float(signals["scam_composite_score"])
        vag = float(signals["vagueness_score"])
        tlen = float(np.log1p(signals["text_length"]))
        clen = float(np.log1p(len(company_profile or "") if use_explicit_metadata else (400 if "Company Profile / Background" not in signals["missing_sections"] else 0)))
        rlen = float(np.log1p(len(requirements or "") if use_explicit_metadata else (400 if "Detailed Role Requirements / Industry" not in signals["missing_sections"] else 0)))
        blen = float(np.log1p(len(benefits or "") if use_explicit_metadata else (300 if "Benefits & Compensation Structure" not in signals["missing_sections"] else 0)))
        has_sal = float(1.0 if (salary_range or signals["salary_baits"]) else 0.0)
        tele = float(signals["telecommuting"])
        logo = float(signals["has_company_logo"])
        ques = float(signals["has_questions"])

        F_num = np.array([[urg, scam, vag, tlen, clen, rlen, blen, has_sal, tele, logo, ques]], dtype=np.float32)

        branch_res = self.predict_proba_branches(
            X_tfidf, F_num, use_structured=use_explicit_metadata
        )
        ml_prob = float(branch_res["final_prob"][0])
        struct_prob = float(branch_res["struct_prob"][0])
        text_prob = float(branch_res["text_prob"][0])
        knn_prob = float(branch_res["knn_prob"][0])
        max_sim = float(branch_res["max_sim"][0])

        # Branch 4: Calibrated Domain & Red-Flag Heuristic Risk Score (0.0 to 1.0)
        heur_points = 0.0
        heur_points_max = 100.0
        # High-risk scam triggers (fees, wire transfers, WhatsApp/Telegram interviews, SSN/bank)
        heur_points += min(45.0, signals["high_scam_count"] * 22.0)
        # Medium-risk triggers (work from home, data entry, no experience, daily/weekly payout)
        heur_points += min(28.0, signals["med_scam_count"] * 9.5)
        # Urgency phrases
        heur_points += min(18.0, signals["urgency_score"] * 6.5)
        # Free email domain (gmail/yahoo/hotmail/etc.)
        if len(signals["free_emails"]) > 0:
            heur_points += 24.0
        # Salary bait ($5000/week, $500/day, etc.)
        if len(signals["salary_baits"]) > 0:
            heur_points += 18.0
        # Vagueness & short length penalty
        if signals["text_length"] < 35:
            heur_points += 22.0
        elif signals["text_length"] < 75 and signals["vagueness_score"] >= 3:
            heur_points += 15.0
        heur_points += signals["vagueness_score"] * 4.5
        if signals["hype_score"] > 5:
            heur_points += min(12.0, (signals["hype_score"] - 3) * 2.0)
        if use_explicit_metadata and has_company_logo == 0:
            heur_points += 10.0

        # Subtract strong legitimacy signals
        legit_deduction = min(38.0, signals["legitimacy_score"] * 5.5)
        if "Company Profile / Background" not in signals["missing_sections"] and signals["text_length"] >= 150:
            legit_deduction += 10.0
        heur_score = np.clip((heur_points - legit_deduction) / 85.0, 0.02, 0.99)

        # Blend ML Ensemble with Domain Calibrator
        # When explicit red flags are present in live text, ensure heuristic calibrator boosts detection
        if signals["high_scam_count"] >= 1 or len(signals["free_emails"]) >= 1 or (signals["med_scam_count"] >= 2 and signals["urgency_score"] >= 1):
            combined_prob = max(
                0.60 * ml_prob + 0.40 * heur_score,
                0.72 * heur_score + 0.28 * ml_prob,
            )
        elif signals["legitimacy_score"] >= 4 and signals["high_scam_count"] == 0 and len(signals["free_emails"]) == 0:
            combined_prob = 0.78 * ml_prob + 0.22 * heur_score
        else:
            combined_prob = 0.72 * ml_prob + 0.28 * heur_score

        combined_prob = float(np.clip(combined_prob, 0.005, 0.995))
        prediction = int(combined_prob >= self.threshold)

        if combined_prob >= 0.60:
            verdict_label = "HIGH RISK — LIKELY FAKE"
            risk_tier = "HIGH"
        elif combined_prob >= self.threshold:
            verdict_label = "MODERATE-HIGH RISK — SUSPICIOUS POSTING"
            risk_tier = "MODERATE"
        elif combined_prob >= 0.25:
            verdict_label = "LOW-MODERATE RISK — PROCEED WITH CAUTION"
            risk_tier = "CAUTION"
        else:
            verdict_label = "LOW RISK — LIKELY REAL"
            risk_tier = "LOW"

        confidence_pct = (
            combined_prob * 100.0 if prediction == 1 else (1.0 - combined_prob) * 100.0
        )

        # Extract top contributing n-grams from the document
        X_csr = X_tfidf.tocsr()
        doc_indices = X_csr.indices
        doc_vals = X_csr.data
        suspicious_tokens = []
        legitimate_tokens = []
        if len(doc_indices) > 0 and self.vocab_items_:
            # Contribution in NBSVM text branch: tf_idf * r * w_text
            contribs = doc_vals * self.r_[doc_indices] * self.w_text_[doc_indices]
            order = np.argsort(contribs)
            for idx_pos in order[::-1][:8]:
                if contribs[idx_pos] > 0.01:
                    tok_name = self.vocab_items_[doc_indices[idx_pos]].replace("_", " ").strip()
                    if not tok_name.startswith("email token") and len(tok_name) > 2:
                        suspicious_tokens.append(
                            {"token": tok_name, "impact": round(float(contribs[idx_pos]), 4)}
                        )
            for idx_pos in order[:8]:
                if contribs[idx_pos] < -0.01:
                    tok_name = self.vocab_items_[doc_indices[idx_pos]].replace("_", " ").strip()
                    if len(tok_name) > 2:
                        legitimate_tokens.append(
                            {"token": tok_name, "impact": round(float(abs(contribs[idx_pos])), 4)}
                        )

        # Similar historical postings from k-NN memory bank
        similar_cases = []
        knn_idxs = branch_res["knn_indices"][0]
        knn_sims = branch_res["knn_sims"][0]
        sim_order = np.argsort(knn_sims)[::-1][:3]
        for pos in sim_order:
            tr_i = int(knn_idxs[pos])
            sim_val = float(knn_sims[pos])
            if sim_val > 0.08:
                similar_cases.append(
                    {
                        "title": self.train_titles_[tr_i] if tr_i < len(self.train_titles_) else f"Posting #{tr_i}",
                        "label": "FAKE" if int(self.y_train_[tr_i]) == 1 else "REAL",
                        "similarity_pct": round(sim_val * 100.0, 1),
                    }
                )

        # Build human-readable explanations
        reasons = []
        positive_notes = []

        if signals["high_scam_count"] > 0:
            reasons.append(
                f"Critical scam indicators detected ({signals['high_scam_count']} match(es)): {', '.join(signals['matched_high_scam'][:5])}"
            )
        if signals["med_scam_count"] > 0:
            reasons.append(
                f"High-risk job-lure phrases found ({signals['med_scam_count']} match(es)): {', '.join(signals['matched_med_scam'][:5])}"
            )
        if len(signals["free_emails"]) > 0:
            reasons.append(
                f"Uses personal/free email address for recruitment: {', '.join(signals['free_emails'][:3])}"
            )
        if len(signals["salary_baits"]) > 0:
            reasons.append(
                f"Contains aggressive short-term pay lure pattern: {', '.join(signals['salary_baits'][:2])}"
            )
        if signals["urgency_score"] > 0:
            reasons.append(
                f"Contains {signals['urgency_score']} high-pressure urgency phrase(s): {', '.join(signals['matched_urgency'][:4])}"
            )
        if signals["vagueness_score"] >= 3:
            reasons.append(
                f"High structural vagueness ({signals['vagueness_score']}/4 core sections missing: {', '.join(signals['missing_sections'])})"
            )
        elif signals["vagueness_score"] == 2:
            reasons.append(
                f"Moderate structural vagueness (missing: {', '.join(signals['missing_sections'])})"
            )
        if signals["hype_score"] > 4:
            reasons.append(
                f"Excessive promotional hype ({signals['exclamation_count']} exclamation marks, {signals['caps_word_count']} ALL-CAPS words)"
            )
        if signals["text_length"] < 40:
            reasons.append(
                f"Posting is unusually short and lacks substantive role details ({signals['text_length']} words)"
            )
        if suspicious_tokens and prediction == 1:
            top_tok_str = ", ".join(f"'{t['token']}'" for t in suspicious_tokens[:4])
            reasons.append(f"NLP model flagged high-risk vocabulary patterns: {top_tok_str}")

        if signals["legitimacy_score"] > 0:
            positive_notes.append(
                f"Contains {signals['legitimacy_score']} corporate/professional standard markers ({', '.join(signals['matched_legitimacy'][:5])})"
            )
        if signals["vagueness_score"] <= 1 and signals["text_length"] >= 120:
            positive_notes.append(
                f"Well-structured posting ({signals['text_length']} words) with clear role, qualification, and company details"
            )
        if legitimate_tokens:
            leg_tok_str = ", ".join(f"'{t['token']}'" for t in legitimate_tokens[:4])
            positive_notes.append(f"Strong legitimate domain vocabulary: {leg_tok_str}")

        if not reasons:
            if prediction == 0:
                reasons.append("No major linguistic, financial, or structural fraud red flags detected.")
            else:
                reasons.append("Statistical n-gram distribution closely matches known fraudulent postings in the EMSCAD dataset.")

        return {
            "prediction": prediction,
            "verdict_label": verdict_label,
            "risk_tier": risk_tier,
            "fraud_probability": round(combined_prob, 4),
            "confidence_pct": round(confidence_pct, 1),
            "risk_score": round(combined_prob * 100.0, 1),
            "branch_scores": {
                "nbsvm_text_prob": round(text_prob * 100.0, 1),
                "nbsvm_structured_prob": round(struct_prob * 100.0, 1),
                "knn_similarity_prob": round(knn_prob * 100.0, 1),
                "heuristic_calibrator_prob": round(float(heur_score) * 100.0, 1),
                "max_historical_similarity_pct": round(max_sim * 100.0, 1),
            },
            "signals": signals,
            "reasons": reasons,
            "positive_notes": positive_notes,
            "suspicious_tokens": suspicious_tokens,
            "legitimate_tokens": legitimate_tokens,
            "similar_cases": similar_cases,
        }


def save_artifact(obj, path: str):
    """Saves model or vectorizer using standard pickle Protocol 4 (compatible with joblib & pickle)."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Ensures full cross-compatibility between NumPy 1.x (numpy.core) and NumPy 2.x (numpy._core)."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = module.replace("numpy._core", "numpy.core", 1)
        elif module.startswith("numpy.core"):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = module.replace("numpy.core", "numpy._core", 1)
        return super().find_class(module, name)


def load_artifact(path: str):
    """Loads model or vectorizer via cross-version compatible pickle or joblib."""
    try:
        with open(path, "rb") as f:
            return _NumpyCompatUnpickler(f).load()
    except Exception:
        import joblib
        return joblib.load(path)
