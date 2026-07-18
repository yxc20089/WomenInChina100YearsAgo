# HunyuanOCR 1.5 targeted page suite

Date: 2026-07-18

Source: volume 219, page 308, corrected-polarity native-resolution render

Status: model comparison and visual adjudication; not historian gold

## Method

Eleven byte-preserved crops were taken from the same 6176×8960 page render.
HunyuanOCR 1.5 used the official `structured_parse` task, prompt
`提取图中的文字。`, greedy decoding, and three repeats per crop. All three
repeats were identical in every case. PP-OCRv6 was run once on each identical
crop. Neither model output was normalized before comparison.

The adjudication distinguishes three layers:

1. raw model output;
2. visually supported transcription in image-coordinate order;
3. reading-order or contextual interpretation.

Context can rank hypotheses, but it cannot silently replace an unclear glyph.

## Results

| ID | Purpose | HunyuanOCR 1.5 raw output | PP-OCRv6 evidence | Adjudication |
|---|---|---|---|---|
| C01 | stylized RTL venue | `中央大戲院` | `院默大央中` (0.822) | Hunyuan exact: `中央大戲院` |
| C02 | second RTL venue | `卡爾登影戲院` | `院戲影登爾卡` (0.996) | Both support `卡爾登影戲院`; Paddle needs reversal |
| C03 | third RTL venue | `奧迪安大戲院` | `院戲大安迪奧` (0.943) | Both support `奧迪安大戲院`; Paddle needs reversal |
| C04 | clean RTL date | `二月十五日期` | `期星日五十月二` (0.998) | Hunyuan omitted `星`; supported reading is `二月十五日星期` |
| C05 | small RTL sentence | `片新换更日期星及四期星逢每院本意注，` | relevant line `片新换更日期星及四期星逢每院本意注` (0.966) | Both expose image order; supported reading is `注意本院每逢星期四及星期日更換新片`; both simplify `換` in raw output |
| C06 | Latin LTR control | `(ERNST LUBITSCH)` / `# KISS ME AGAIN` | `(ERNST LUBTTSCH)` (0.952), `KISS ME AGAIN` (0.963) | Hunyuan glyphs exact but inserts Markdown `#` |
| C07 | disputed vertical column | `茶餘酒後。對着嬌兒慈母。愛能情人。說用齊市` | low-confidence conflicting strings | Human review corrects the phrase to `愛寵情人。`; Hunyuan substituted `能` for `寵`; the final segment remains unresolved |
| C08 | degraded vertical column | `觀這片。不僅悅且娛心。還可增長見識。因為高` | noisy, but contains `悅目` | Human review confirms `不僅悅目娛心`; Hunyuan substituted `且` for `目`; the crop ending is incomplete |
| C09 | clean long vertical text | see transcript below | three noisier columns, including `英皇時召雷臨宮中` | Contextual human review resolves `英皇時召彼臨宮中`: `彼` is the object of `召`, while `臨宮中` is the verb phrase; neighboring text identifies `霍爾平（Hans Holbein）`, not `霍臨` |
| C10 | dense degraded line | incoherent multi-line prose | three low-confidence incoherent lines | Reject both; Hunyuan output is not safely grounded |
| C11 | blank control | `图中没有文字` | no regions | Correct abstention; response language is Simplified Chinese |

### C09 Hunyuan raw transcript

```text
較上次更受歡迎、英皇時召霍臨宮中、與之談話、並請
 其作畫、居約六七年、彼離英而遊歷、先至比利士之白
 魯捨爾、大受當地藝術界之殷待、後重至白塞爾、少居
```

The reviewed first-line phrase is `英皇時召彼臨宮中`: `彼` is the object of
`召`, and `臨宮中` means coming to the palace. Neighboring columns identify
the subject as `霍爾平（Hans Holbein）`; `霍臨` is therefore not a person name.
The raw Hunyuan `霍` and Paddle `雷` readings remain preserved.

## Human checks requested

- **C07:** the reviewed phrase is `愛寵情人。`; transcribe only the unresolved
  column ending if it becomes legible from another scan or source.
- **C08:** resolved as `不僅悅目娛心`; the text after the crop boundary is not
  claimed by this test.
- **C09:** resolved as `英皇時召彼臨宮中` using the glyph, grammar, pronoun
  chain, and adjacent identification of `霍爾平（Hans Holbein）`.

Review crops:

- [C07 original](../../artifacts/ocr-challenger/suite-v219-p0308/C07_vertical_disagreement.png)
  and [4× review image](../../artifacts/ocr-challenger/suite-v219-p0308/C07_main-4x.png)
- [C08 original](../../artifacts/ocr-challenger/suite-v219-p0308/C08_vertical_degraded.png)
  and [4× review image](../../artifacts/ocr-challenger/suite-v219-p0308/C08_main-4x.png)
- [C09 original](../../artifacts/ocr-challenger/suite-v219-p0308/C09_vertical_clean.png)
  and [4× review image](../../artifacts/ocr-challenger/suite-v219-p0308/C09_right-4x.png)
- [C09 adjacent Holbein identification](../../artifacts/ocr-challenger/suite-v219-p0308/C09_holbein_context.png)
  and [3× review image](../../artifacts/ocr-challenger/suite-v219-p0308/C09_holbein_context-3x.png)

## Decision

HunyuanOCR 1.5 is the leading difficult-region recognizer from this test, not
a page-wide single-source authority. Use Paddle for fast detection, polygons,
and an independent recognition hypothesis. Escalate stylized, low-confidence,
reading-order-sensitive, or disagreeing crops to Hunyuan. Preserve every raw
hypothesis and require review for consequential disagreement or unsupported
contextual repair.

The suite rejects unconditional RTL reversal: C04 and C05 need reversal, while
C06 must stay LTR and vertical columns need layout reconstruction. Reading
direction therefore belongs to the layout layer and must be recorded per text
line or region.

## C07 correction history

The project owner reviewed the enlarged source crop and supplied `爱宠情人。`.
The source-script transcription is recorded as `愛寵情人。`. This supersedes
the report's earlier contextual hypothesis `愛能解人意`, which was incorrect.
It does not alter either model's raw output, and it is not labeled independent
two-review gold. The unreadable final segment remains explicitly unresolved.

## C08 and C09 correction history

The project owner confirmed C08 as `不僅悅目娛心`. For C09, the initial visual
hypothesis `霍` and possible segmentation `霍臨` were checked against the
adjacent columns. Those columns name `霍爾平（Hans Holbein）`, while the target
sentence continues with `與之談話、並請其作畫` and later `彼離英`. The
reviewed reading is therefore `英皇時召彼臨宮中`, with `臨宮中` functioning
as the verb phrase. These are single-project-review corrections, not
independent two-review benchmark gold.
