# Proposal: target-based rejection thresholds for LibEMG

This folder holds a self-contained module (`rejection.py`, numpy only) and its
tests (`test_rejection.py`), written to drop into LibEMG as
`libemg/rejection.py` without changing any existing API. The tests check the
misfire curve output for output against `EMGClassifier.add_rejection()`.

Order of operations:

1. Open the GitHub issue below on `LibEMG/libemg` and wait for a maintainer to
   say whether they want it, and where it should live.
2. Then fork, add `rejection.py` and the tests, add a short docs page, and open
   the PR with the description below.
3. Send the email after the issue is up, linking to it.

---

## GitHub issue

**Title:** Choosing the rejection threshold from a target error rate (conformal + adaptive)

**Body:**

`add_rejection()` takes a fixed threshold (default 0.9). The error rate that
threshold produces varies a lot across users and conditions. With
user-specific LDA models on the CIIL electrode-shift data, 0.9 moved the
device unintentionally on 44% of decisions after the 45° rotation; on the
Fougner limb-position data, 18.8%.

I'd like to contribute two small helpers that pick the threshold from a target
instead, e.g. "unintended movements on at most 5% of decisions":

- `conformal_rejection_threshold(people, target, rest_class)`: conformal risk
  control with people as the exchangeable unit. If a new user is exchangeable
  with the calibration users, their expected misfire rate is at most the
  target. Pure function; the result goes straight into `add_rejection()`.
- `AdaptiveRejection(target, start, eta, feedback_rate)`: updates the threshold
  online from a misfire bit (e.g. during a cued task or alongside
  context-informed incremental learning). Deterministic per-user bound on any
  decision sequence; an importance-weighted variant keeps the bound in
  expectation when only a random share of decisions get feedback.

A misfire here is a non-rest output that differs from the intended class, so a
confident "rest" is never counted as one.

Evaluation on UCI, 3DC, CIIL and Fougner (91 people, leave-one-person-out):
retraining every 5 s plus the adaptive threshold kept every user at or below
the target at all six levels from 3% to 20% on all four datasets; retraining
with 0.9 left 44 of 91 users above 5%. Code, results and a paper draft:
https://github.com/kishoren18-hue/steadyhand

The proposed module is about 100 lines, numpy only, touches no existing code,
and its tests compare against `EMGClassifier`'s own rejection output. Would you
want this in LibEMG? If so, is `libemg/rejection.py` the right place, or would
you prefer it on `EMGClassifier` (e.g. `add_rejection(target=...)`)? Happy to
adapt to whatever fits the library.

---

## Pull request description (after a maintainer agrees)

**Adds `libemg.rejection`: rejection thresholds chosen from a target misfire rate**

- `misfires(predictions, labels, rest_class=None)`
- `misfire_curve(probabilities, misfire, thresholds)`
- `conformal_rejection_threshold(people, target=0.05, rest_class=None)`
- `AdaptiveRejection(target=0.05, start=0.9, eta=0.5, feedback_rate=1.0)`

No changes to existing modules; numpy only. Tests:
`misfire_curve` matches `EMGClassifier.add_rejection()` output for output at
several thresholds; the conformal threshold keeps the expected misfire rate
below target on simulated users; the adaptive bound holds on sequences with an
abrupt shift. Docs page included with a worked example.

References: Angelopoulos et al., Conformal risk control, ICLR 2024; Gibbs and
Candès, Adaptive conformal inference under distribution shift, NeurIPS 2021.

---

## Email to the maintainers

**To:** Ethan Eddy (LibEMG maintainer, UNB). Find his current address on his
UNB or GitHub profile; cc Evan Campbell if he is listed as co-maintainer.

**Subject:** LibEMG contribution: rejection thresholds with a misfire-rate guarantee

Hi Ethan,

I built SteadyHand on top of LibEMG: it chooses the rejection threshold from a
target misfire rate instead of a fixed number, using conformal risk control
across users and an adaptive threshold updated from feedback.

On your CIIL electrode-shift data and three other public datasets (91 people),
retraining every 5 seconds plus the adaptive threshold kept every user at or
below the target at every level from 3% to 20%, while retraining with 0.9 left
44 of 91 users above 5%. The finding I'd most like your view on is that a
threshold alone couldn't rescue control after the shift; it needed retraining
of the kind your context-informed incremental learning work does.

I opened an issue proposing a small, dependency-free module for LibEMG:
[issue link]. Code and a paper draft are here:
https://github.com/kishoren18-hue/steadyhand

Two questions, if you have time: would this fit in LibEMG, and do you see a
flaw in treating a non-rest wrong output as the error to control? I'm planning
a live test with a MindRove or OYMotion armband next and would welcome
any advice on protocol.

Thanks for LibEMG and the datasets,
Kishore Nuthalapati
https://kishorenuthalapati.com
