## Why this section? 
[17/07/2026]
This is currently an workshop level paper and it has to adress limitations and accept further modification until it becomes a truely valuable resources to community.So, I am going to be honest about it. Point our exact potential as well as drawbacks and hopefully there'll be better days!

Since I am a little occupied, I will start with reviewers comments which I am willing to addrsss in later version of my paper.
[https://openreview.net/forum?id=803uxwFQep](https://openreview.net/forum?id=803uxwFQep)
```
Clean engineering contribution with strong empirical results - Accept
Official Review by Reviewer yzkT 10 Jun 2026, 07:26 (modified: 11 Jun 2026, 10:58)
Review:
Summary

The paper addresses a well-defined failure mode in cross-lingual ASR: when a compact, English-optimized model (Moonshine, 61.5M parameters) is applied to Bengali, its byte-level tokenizer
fragments Bengali words into long token chains (fertility =9.16), causing catastrophic autoregressive collapse at inference time despite stable training loss. The proposed fix is a three-phase
"tokenizer transplantation" pipeline: (1) fine-tune the acoustic encoder with the original tokenizer to adapt Bengali acoustic representations; (2) surgically replace the decoder's embedding matrix
with a native BanglaBERT SentencePiece vocabulary (reducing fertility to = 1.30); (3) run a two-stage recovery fine-tuning to re-align the randomly initialized decoder embeddings to the
already-adapted encoder. Evaluated on the 882-hour Lipi-Ghor dataset, the transplanted model achieves 21.54% WER and 10.79% CER - the lowest CER among all baselines - while running at
RTF 0.0053, 3.5x faster than optimized Whisper-Medium deployments.

Strengths

The problem diagnosis is precise and well-motivated: the fertility gap between English and Bengali tokenization is quantified clearly, and the causal link to autoregressive collapse is explained
convincingly. The three-phase pipeline is technically sound and the decoupling of acoustic adaptation from vocabulary alignment is a genuinely useful insight that extends to other
morphologically rich, non-Latin script languages. The empirical results are strong: achieving competitive WER with 10x fewer parameters than Whisper-Medium and the best CER overall is a
meaningful result. The paper is well-written and the methodology is reproducible (code provided). The comparison table covers a sensible range of zero-shot and fine-tuned baselines.

Weaknesses

The paper evaluates only on the Lipi-Ghor dataset; it is unclear whether the transplantation approach generalizes to other Bengali corpora or to other morphologically rich languages beyond
Bengali. The two-stage recovery training schedule (high LR then low LR) is presented as a design choice but there is no ablation study justifying the specific hyperparameters or comparing
against alternative recovery strategies (e.g., warm-up schedules, gradual unfreezing). The comparison with Faster Whisper Medium is somewhat unfair in the parameter count framing -
Whisper-Medium benefits from vastly more pre-training data, so the parameter efficiency claim should be qualified. The paper is short (5 pages) and could benefit from analysis of failure cases or
error type breakdowns.
Rating: 7: Good paper, accept
Confidence: 3: The reviewer is fairly confident that the evaluation is correct
``` 

and Here comes next,
```
Useful tokenizer-transplantation method with a strong efficiency result, undercut by an overstated accuracy claim and no
ablations
Official Review by Reviewer rrMG
Review:
This paper adapts the compact Moonshine ASR model to Bengali by replacing its high-fertility English byte-level tokenizer with a native-script vocabulary, plus a two-stage recovery schedule. The
efficiency result is genuinely useful, but the paper overstates its accuracy result and provides no ablations.
Strengths

. Sensible, practical idea: fixing tokenizer fertility rather than scaling the model, with a clean fertility reduction (9.16 to 1.30) that directly explains the 85.8% sequence-length drop.
. Strong efficiency result: about 61.5M parameters matching a ~769M model on CER with a much better real-time factor, which matters for edge deployment.
. More reproducible than most short papers (optimizer, learning rates, epochs, mixed precision, anonymized repo).

Weaknesses

. The "state-of-the-art WER" claim is contradicted by the paper's own Table 2: Faster Whisper Medium reaches 21.28% vs. the proposed 21.54%. The model leads on CER, not WER. The abstract
also reports 20.4% against the table's 21.54%.
. No ablations isolating the three pipeline components (pre-transplant acoustic fine-tuning, transplant, two-stage recovery), so each is asserted necessary rather than shown. "Autoregressive
collapse" is never defined or measured.
. The tokenizer is mis-described: the abstract says WordPiece, Section 4.2 says SentencePiece and cites Kudo & Richardson, but BanglaBERT uses WordPiece. The evaluation surface is also
muddled (different test sets for WER/CER and RTF, unclear decoding configuration, and unstated hardware for the reported RTF).
Rating: 6: Marginally above acceptance threshold
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct

01 Jun 2026, 14:59 (modified: 11 Jun 2026, 10:58)
```
