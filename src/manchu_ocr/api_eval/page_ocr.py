"""Whole-page API inputs and alignment-based transcription metrics."""
from __future__ import annotations

import re
from pathlib import Path

from manchu_ocr.api_eval.zero_shot import RecognitionSample, _sample_id, aggregate_strict_metrics
from manchu_ocr.metrics.recognition_metrics import levenshtein_error_counts


def load_page_manifest(path):
    # Lazy imports keep word-only API use independent of model dependencies.
    from manchu_ocr.data.page_splits import read_manifest
    from manchu_ocr.metrics.e2e_ocr_metrics import parse_page_annotation
    from manchu_ocr.pipelines.full_ocr_pipeline import FullOCRPipeline

    samples, seen = [], set()
    for image, annotation in read_manifest(path):
        if image in seen:
            raise ValueError(f"Duplicate page in manifest: {image}")
        seen.add(image)
        items = parse_page_annotation(annotation)
        if not items or any(not item["text"] for item in items):
            raise ValueError(f"Page needs complete word transcriptions: {annotation}")
        text = FullOCRPipeline.build_page_text(FullOCRPipeline.group_columns(items))
        index = len(samples)
        # Include image bytes in the identity: replacing an image invalidates API cache.
        from manchu_ocr.api_eval.zero_shot import manifest_sha256
        identity = _sample_id(index, image, text + manifest_sha256(image))
        samples.append(RecognitionSample(index, image, text, identity))
    if not samples:
        raise ValueError("Page manifest is empty")
    return samples


def parse_page_response(text):
    """Reject explicit refusals/structured output; never extract a likely answer."""
    text = str(text).strip()
    if (not text or "```" in text or any(c in text for c in "<>{}[]") or re.search(r"^\s*#", text, re.M)
            or re.search(r"\b(i am sorry|i'm sorry|i cannot|i can't|unable to|cannot transcribe|cannot provide)\b", text, re.I)
            or re.search(r"[^a-zA-Z\s'’ʼūščžŋ.,:;!?()\-]", text)):
        return ""
    return text


def aggregate_page_metrics(references, predictions):
    from manchu_ocr.metrics.e2e_ocr_metrics import aggregate_edit_counts, page_transcription_metrics
    if not references or len(references) != len(predictions):
        raise ValueError("Need equally sized, nonempty page lists")
    char_metrics = aggregate_strict_metrics(
        [re.sub(r"\s+", "", t) for t in references],
        [re.sub(r"\s+", "", t) for t in predictions])
    word_s = word_d = word_i = word_n = 0
    for ref, pred in zip(references, predictions):
        s, d, i = levenshtein_error_counts(pred.split(), ref.split())
        word_s += s
        word_d += d
        word_i += i
        word_n += len(ref.split())
    wa = (word_n - word_s - word_d) / max(word_n, 1)
    pairs = list(zip(predictions, references))
    page_counts = aggregate_edit_counts(pairs)
    char_metrics.update(
        num_pages=len(references), word_accuracy=wa, exact_word_accuracy=wa, WA=wa,
        word_substitutions=word_s, word_deletions=word_d, word_insertions=word_i,
        reference_words=word_n, word_error_rate=(word_s + word_d + word_i) / max(word_n, 1),
        page_transcription_cer=page_counts["cer"],
        page_transcription_score=sum(page_transcription_metrics(p, r)["page_transcription_score"] for p, r in pairs) / len(pairs),
        page_exact_accuracy=sum(p == r for p, r in pairs) / len(pairs),
        word_accuracy_definition="correct reference tokens in minimum-edit word alignment / reference tokens; no spatial matching",
    )
    return char_metrics
