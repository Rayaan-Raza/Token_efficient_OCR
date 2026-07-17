# 03 — Success gates

Evaluated on DocVQA **n=1000**, method stamp `raven_select` **1.0.0** with
full-page reader replaced.

| Outcome | Condition |
|---------|-----------|
| **PASS** | RAVEN-Select+OCR-SeamResize significantly beats **resize** and **original RAVEN-Select** on ANLS (both CI lower bounds > 0) |
| **PARTIAL** | Significantly beats resize but not original RAVEN-Select |
| **FAIL** | Otherwise |

Secondary reporting (not the gate):

- Does OCR-SeamResize alone beat Resize?
- Does MarginCrop or WhitespaceCompress explain most of the gain?
- Failure cases where compression damages text/layout

## Non-claims

- Tiny mean gains with CI crossing zero do **not** rescue a strong method paper.
- Transfer is out of scope until DocVQA gate is PASS or PARTIAL with clear
  reader-alone gains.
- Do not claim standard MP-DocVQA from contact sheets.
