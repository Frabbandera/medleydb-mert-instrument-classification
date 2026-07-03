# 02 - Theory: frozen MERT for isolated-stem classification

This is the conceptual document. It explains what was implemented and why; it
does not contain environment setup commands. For execution, use the
[local CPU/GPU guide](commands.md) or the [Google Colab guide](google_colab.md).

## Task

Given a five-second mono segment from one MedleyDB stem, predict one coarse instrument family. This is single-label segment classification: the model outputs one of the selected balanced classes, not a list of instruments in a mixture.

The course brief permits either isolated-stem or polyphonic classification and asks for an end-to-end, reproducible evaluation. Isolated stems are the controlled first condition. They let us determine whether the metadata, timbre representation, and classifier work before adding overlapping instruments.

## Why MERT

MERT is a self-supervised acoustic music model. Its convolutional frontend and Transformer encoder were pretrained using acoustic and musically informed targets, including a constant-Q-based teacher. The 95M-parameter checkpoint returns a sequence of 768-dimensional hidden representations for 24 kHz audio.

This project uses the final hidden layer and averages it over time. The resulting vector summarizes a segment and is passed to either a linear classifier or the default `Linear-ReLU-Dropout-Linear` head. The MERT parameters are frozen and extraction runs under `torch.inference_mode()`.

Freezing is a practical experimental choice:

- it fits comfortably on an 8 GB laptop GPU with batch size one;
- it prevents a small debug subset from overfitting a large backbone;
- it separates audio/representation problems from classifier training;
- it makes repeated classifier experiments inexpensive once embeddings are cached.

Caching also fixes the representation used by every classifier run. Each cache records the model name/revision, layer, pooling rule, sample rate, and a fingerprint of the segment table. A cache is reused only when these fields match.

## Why coarse balanced classes

MedleyDB's detailed instrument labels are long-tailed. A classifier trained directly on rare labels would have classes represented by one song, making a track-disjoint test split impossible. The first taxonomy groups exact official labels into vocals, guitar, bass, drums, keyboards, strings, winds/brass, percussion, and electronic sounds. Ambiguous labels remain `other_unknown` and are not trained.

The script ranks eligible families by the number of distinct tracks, not by the duration of a few long songs. It then chooses a common number of non-silent segments for every retained class. Macro-F1 is therefore the checkpoint and headline metric; accuracy and weighted-F1 are also reported.

## Why track-level splitting

Adjacent segments from one stem share performer, production, room, microphone, and musical content. Random segment splitting would let nearly identical audio appear in training and testing and would inflate performance. The subset builder assigns each `track_id` once, then derives every segment split from that mapping. Its report explicitly computes all pairwise track-set intersections; they must be zero.

## Boundaries and future work

The Lead Instrument Detection system by Ou et al. is useful architectural evidence for five-second MedleyDB segmentation, MERT processing, Lightning training, and multitrack evaluation. Its target is different: it identifies the perceptually leading track over time. This repository predicts the family of an isolated stem and does not copy its multitrack attention model.

The Persian-instrument study demonstrates a later direction: construct culturally and temporally plausible mixtures, aggregate MERT layers, and train sigmoid outputs with binary cross-entropy. Those choices are appropriate for multi-label polyphonic audio, not for this first single-label isolated-stem experiment.

## References

1. Politecnico di Milano. *Selected Topics in Music and Audio Engineering:
   Research Project*, course brief, 15 May 2026. Supplied separately to course
   participants.
2. R. Bittner et al. “MedleyDB: A Multitrack Dataset for Annotation-Intensive
   MIR Research.” *Proceedings of ISMIR*, 2014. See the
   [MedleyDB project](https://medleydb.weebly.com/).
3. Y. Li et al. “MERT: Acoustic Music Understanding Model with Large-Scale
   Self-Supervised Training.” *ICLR*, 2024.
   [arXiv:2306.00107](https://arxiv.org/abs/2306.00107).
4. L. Ou, Y. Takahashi, and Y. Wang. “Lead Instrument Detection from Multitrack
   Music.” 2025. [arXiv:2503.03232](https://arxiv.org/abs/2503.03232).
5. D. H. Esfangereh et al. “Persian Musical Instruments Classification Using
   Polyphonic Data Augmentation.” 2025.
   [arXiv:2511.05717](https://arxiv.org/abs/2511.05717).
