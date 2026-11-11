# Multi-stream MQ decoding on one core: experiment record

2026-11-11. Question: Tier-1 time is the MQ arithmetic decoder, and every
decision depends on the one before it. Code blocks are independent of each
other, so can one core decode two or four of them at once, a little of each
in turn, and overlap their dependency chains?

Outcome in one paragraph: in isolation, yes, by a lot; in the decoder, no.
Replaying real context traces, two interleaved decoders run 1.3 to 1.5 times
as fast as one, but only with a decoder that has no data-dependent branch
and only when the streams alternate every one or two symbols. A Tier-1 that
alternates two blocks every stripe column, which is as fine as it can go
without becoming a per-symbol state machine, decodes whole images 3 to 8 %
slower than one block at a time (4 to 10 % slower with four workers). The
prototype is kept here as a patch and is not in the decoder. What the
experiment did give the decoder is the branch-free decision itself: used
for the decisions Tier-1 cannot predict, it makes single-threaded decode
12 to 14 % faster on lossless streams and 7 to 10 % on lossy ones.

## What is here

| File | What |
|---|---|
| `src/bin/bench.rs` | Replay benchmark. Compiles the production `src/mq/mod.rs` by path. |
| `record.patch` | Adds an `mq-trace` cargo feature and a `record` binary that dumps every MQ segment's bytes, contexts and decoded bits. |
| `tier1-lanes-prototype.patch` | The resumable two-block Tier-1 that was measured and not adopted. |
| `replay-results-2026-11-11.txt` | Full output of the replay runs summarised below. |

Both patches apply from the repository root to the commit that added this
directory (`git apply codecs/jp2lam/decode-fix-plan/mq-interleave-experiment/record.patch`).

## Method

1. **Traces.** With `record.patch`, a serial decode records for each code
   block its compressed bytes, the context of every MQ decision and the bit
   decoded. Three streams, all OpenJPEG-encoded 1024 x 1024 crops with 64 x 64
   blocks: a lossless 5/3 gray scan (7.27 M decisions, 0.79 bits each), a
   lossy 9/7 RGB photograph (3.09 M, 0.54) and a lossy 9/7 gray scan
   (0.89 M, 0.47).
2. **Replay.** `bench` feeds those contexts to the production `MqDecoder`,
   one block after another or with N decoders interleaved, K decisions per
   decoder per turn. Every decoded bit is compared with the recording and
   every decoder's final state with the sequential run's, in every mode.
3. **Context models.** A trace hands the decoder its next context for free;
   Tier-1 does not. Three models bracket it: *free* (context read from the
   trace), *dependent* (context loaded through a table indexed by the
   previous bit, as flag update plus context lookup does), and *branchy*
   (dependent, plus a conditional store taken when the bit is 1, the branch
   Tier-1 makes on every significance decision).
4. **Two decoders.** `decode_with_ctx` tests for "MPS, no renormalisation"
   first; `decode_unpredictable` runs one branch-free sequence.
5. **Prototype.** `tier1-lanes-prototype.patch` makes block decoding
   resumable after any stripe column (the three passes become per-column
   kernels; a `Lane` holds pass, stripe and column) and advances two blocks
   a column each in turn, in nine specialised pair loops, with one block at
   a time as the fallback for blocks with several codeword segments.

Machine: Intel Xeon (Sapphire Rapids class, about 2.5 GHz), one pinned
virtual core, no hardware counters. Timings are the best of 11 repetitions;
run-to-run noise is around 4 %.

## Replay results

Nanoseconds per decision; in brackets, speed relative to one decoder.

Lossless gray scan:

| Contexts | Decoder | 1 | 2 lanes | 4 lanes | 2 lanes, 2 per turn | 2 lanes, 4 per turn |
|---|---|---|---|---|---|---|
| free | fast path | 7.24 | 7.41 (0.98) | 7.06 (1.03) | 7.54 (0.96) | 7.82 (0.93) |
| free | branch-free | 6.18 | 4.63 (1.34) | 4.63 (1.34) | 5.02 (1.23) | 5.66 (1.09) |
| dependent | fast path | 9.63 | 8.20 (1.17) | 8.30 (1.16) | 9.10 (1.06) | 9.61 (1.00) |
| dependent | branch-free | 8.06 | 5.58 (1.44) | 5.25 (1.54) | 5.94 (1.36) | 7.07 (1.14) |
| branchy | fast path | 11.70 | 11.82 (0.99) | 12.81 (0.91) | 12.98 (0.90) | 13.17 (0.89) |
| branchy | branch-free | 10.20 | 9.07 (1.13) | 9.49 (1.07) | 10.05 (1.01) | 10.44 (0.98) |

Lossy RGB photograph:

| Contexts | Decoder | 1 | 2 lanes | 4 lanes | 2 lanes, 2 per turn | 2 lanes, 4 per turn |
|---|---|---|---|---|---|---|
| free | fast path | 5.80 | 6.28 (0.92) | 6.61 (0.88) | 6.57 (0.88) | 6.61 (0.88) |
| free | branch-free | 6.14 | 4.23 (1.45) | 4.13 (1.49) | 4.52 (1.36) | 5.09 (1.21) |
| dependent | fast path | 8.44 | 7.58 (1.11) | 7.54 (1.12) | 8.16 (1.03) | 8.57 (0.98) |
| dependent | branch-free | 7.95 | 5.33 (1.49) | 4.84 (1.64) | 5.69 (1.40) | 6.67 (1.19) |
| branchy | fast path | 9.68 | 9.31 (1.04) | 9.70 (1.00) | 10.09 (0.96) | 10.62 (0.91) |
| branchy | branch-free | 9.28 | 7.45 (1.25) | 7.51 (1.24) | 8.11 (1.14) | 8.87 (1.05) |

Lossy gray scan:

| Contexts | Decoder | 1 | 2 lanes | 4 lanes | 2 lanes, 2 per turn | 2 lanes, 4 per turn |
|---|---|---|---|---|---|---|
| free | fast path | 5.44 | 5.98 (0.91) | 6.16 (0.88) | 6.18 (0.88) | 6.19 (0.88) |
| free | branch-free | 6.14 | 4.15 (1.48) | 4.06 (1.51) | 4.44 (1.38) | 5.00 (1.23) |
| dependent | fast path | 8.26 | 7.19 (1.15) | 7.28 (1.13) | 8.14 (1.01) | 8.20 (1.01) |
| dependent | branch-free | 7.96 | 5.24 (1.52) | 9.10 (0.88) | 5.66 (1.41) | 6.58 (1.21) |
| branchy | fast path | 9.04 | 8.37 (1.08) | 8.58 (1.05) | 8.90 (1.02) | 9.24 (0.98) |
| branchy | branch-free | 8.83 | 6.83 (1.29) | 6.82 (1.29) | 7.48 (1.18) | 8.31 (1.06) |

(The 9.10 is an outlier repetition set; its neighbours bracket it.)

What the tables say:

- **Interleaving needs a branch-free decoder.** With the fast-path decoder a
  mispredicted test throws away the other lane's work in flight along with
  its own, and two lanes are between 12 % slower and 17 % faster. With the
  branch-free one two lanes gain 34 to 52 % when nothing else branches.
- **Two lanes are enough.** Four add little: at about 4.5 ns a decision the
  core is limited by how many instructions it can retire, not by the chain.
- **The gain is gone within a few symbols of batching.** Two per turn keeps
  roughly two thirds of it, four per turn a third or less. The out-of-order
  window does not reach across a longer run of chained decisions.
- **Tier-1's own branches cost most of what is left.** One unpredictable
  branch per decision, which is more than Tier-1 makes, cuts the two-lane
  gain to 13 to 29 %.

## Whole-image results

Paired lanes against one block at a time, same decoder, same driver,
single thread, best of 24 runs alternating the two binaries:

| Stream | One block | Two lanes |
|---|---|---|
| lossless gray scan, 1 MP | 89.9 ms | 92.8 ms (+3.2 %) |
| lossless RGB photograph, 1 MP | 210.3 ms | 216.4 ms (+2.9 %) |
| lossy RGB photograph, 1 MP | 41.1 ms | 42.5 ms (+3.4 %) |
| lossy gray scan, 1 MP | 10.2 ms | 11.0 ms (+7.8 %) |

With four workers on the 9 to 11 megapixel versions: +4.4 %, +10.2 % and
+4.4 % (lossless gray scan, lossy RGB, lossy gray scan). Earlier in the
experiment, with the fast-path decoder everywhere, the pair loop was 7 to
13 % slower. The driver itself is free: with one lane it matches the plain
loop within 1.5 %. Output is byte-identical in every configuration, serial
and parallel, and the crate's tests pass with the patch applied.

Why it does not carry over: a stripe column is one to three decisions with
some sixty instructions of flag and context work between them and two or
three branches the predictor cannot learn (does this column have anything
to code, did the sample become significant, was that the last one). That is
the "branchy, several per turn" corner of the replay tables, where the gain
is nil, and the pair loop then pays about 3 % more instructions for keeping
two blocks' state live in sixteen registers.

## What was adopted

- `MqDecoder::decode_unpredictable`, the branch-free decision, used for
  refinement bits, signs and significance-pass decisions; the cleanup pass
  and the run-length symbols keep the fast path. A first attempt at this,
  before the experiment, had measured 14 to 20 % slower: the state select
  had been compiled to a jump over the table load. `select_unpredictable`
  keeps it a conditional move.
- The fast-path test reads the registers instead of a stored threshold, and
  the refill adds two plain bytes in one step.

Together, against the commit before: lossless 5/3 streams 12 to 14 % faster
single-threaded, 9/7 streams 7 to 10 %, bit-identical output.

## What would have to be different

Interleaving pays at one or two decisions per turn, so a Tier-1 that used it
would have to switch blocks between decisions, not between columns: each
pass as an explicit state machine resumable mid-column, with its flag work
made branch-free as well. That is a rewrite of the coding passes for a gain
the "branchy" rows put at 13 to 29 % of MQ time at best. The other route
that removes the chain, HTJ2K block coding, changes the codestream and so
does nothing for the files a PDF reader is handed.

## Reproducing

```sh
git apply codecs/jp2lam/decode-fix-plan/mq-interleave-experiment/record.patch
cd codecs/jp2lam/decode-fix-plan/mq-interleave-experiment
cargo build --release
target/release/record some.jp2 some.trace
taskset -c 2 target/release/bench some.trace      # REPS, ONLY, LANES: see bench.rs
```

Without the patch only `bench` builds, and it needs trace files recorded
with it. For the prototype, apply `tier1-lanes-prototype.patch` and flip
`INTERLEAVE_BLOCKS` in `src/decode/t1.rs` to compare one lane with two.
