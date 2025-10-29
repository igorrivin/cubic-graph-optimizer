# When Extended Thinking Isn't Enough: What Frontier AI Models Got Wrong About Mathematical Optimization

> We challenged GPT-5, Gemini 2.5 Deep Think, and Grok 4 Heavy to solve a mathematical optimization problem. Despite 20-90+ minutes of "deep thinking," all three failed—not because the problem was hard, but because extended reasoning without empirical exploration, computational verification, and intellectual honesty leads nowhere.

---

## The Challenge: A Simple Math Problem

Here's what we asked the frontier AI models to do:

**Generate a planar cubic graph (3-regular, embeddable in a plane) that maximizes the number of spanning trees.**

We set two difficulty levels:
- **Easy (N=40 vertices)**: Achieve ln(trees) > 31.28
- **Hard (N=120 vertices)**: Achieve ln(trees) ≥ 95.5

This isn't asking models to prove theorems or make subjective judgments. The spanning tree count is computable via the Matrix-Tree Theorem (a determinant calculation taking milliseconds). Planarity is checkable in linear time. This is a well-defined computational problem with verifiable solutions.

Think of it like asking: "Build me a bridge design that maximizes structural redundancy while staying flat." You can check if it works.

---

## How We Solved It (Spoiler: In 40 Seconds)

Before we dive into the spectacular failures, here's what actually worked:

### Our Approach

1. **Geometric Insight**: Generate random Delaunay triangulations (random points on a sphere, compute convex hull)
2. **Local Optimization**: Apply diagonal flips to maximize spanning trees
3. **Massive Parallelization**: Run 500 independent restarts on 64 CPU cores
4. **Rigorous Verification**: Check every constraint (cubic, planar, connected) and compute exact values

### Results

| Size | ln(trees) | vs Random | Time |
|------|-----------|-----------|------|
| N=40 | **31.354** | 2.25× better | 10 seconds |
| N=120 | **95.574** | 3.81× better | 40 seconds (64 cores) |

The N=40 solution converged to the same optimum 26% of the time (robust basin). The N=120 solution? Only 0.4% convergence (2 out of 500 restarts)—a much rougher landscape requiring broad exploration.

Now let's watch three frontier models spend up to 1.5 hours and fail spectacularly.

---

## GPT-5: The Honeycomb That Wasn't Even Cubic

**Thinking Time**: 20+ minutes (exceeded the normal 15-minute API limit)

**What We Saw**: GPT-5's real-time thinking was fascinating. It visibly cycled through strategies:
- "Creating Halin graphs" (tree skeleton + outer cycle)
- "Ensuring adequate entropy" (realizing it needed randomization!)
- "Creating honeycombs" (regular tiling structures)
- "Navigating mesh lists" (?)

**Final Submission**: A "honeycomb cylinder" construction claiming **ln(trees) ≈ 96.324**

Wait—that would beat our record of 95.574! Extraordinary claim. Let's verify...

### Verification: ❌ INVALID

```python
# What we found:
- Some vertices have degree 5 (should be exactly 3)
- Graph is not planar
- 196 edges instead of 180 (for cubic: 3N/2 = 180)
- Root cause: Edge {1,15} listed for vertex 15 but not vertex 1
```

The adjacency list was constructed by hand and had transcription errors. GPT-5 never ran:
```python
assert all(G.degree(v) == 3 for v in G)  # Would have caught this instantly
```

**The Lesson**: 20 minutes of reasoning about graph structures, but zero seconds of verification. The graph wasn't even cubic, let alone optimal.

---

## Gemini 2.5 Deep Think: The Impossible Fullerene

**Thinking Time**: Fast (a few minutes)

**Strategy**: Pattern-matched to "fullerenes have lots of spanning trees"

**Claim**: "C₄₀ fullerene" with **ln(trees) ≈ 31.899** (that's 74% more than our 31.354!)

Wow, if true, that would crush our baseline. Let's check...

### Verification: ❌ DISQUALIFIED

```python
# What Gemini claimed:
- "C₄₀ fullerene" (convex polyhedron)
- 40 vertices, 60 edges
- 12 pentagons + 10 hexagons (correct counts)

# What we found:
- ✓ Cubic (3-regular)
- ✓ Connected
- ❌ NOT PLANAR

# The critical issue:
By Steinitz's theorem: A graph represents a convex polyhedron
if and only if it is PLANAR and 3-connected.

Gemini's graph is NON-PLANAR → not a valid convex polyhedron
→ NOT ACTUALLY A FULLERENE AT ALL
```

**What went wrong**: Gemini constructed a graph with the right face counts (12 pentagons, 10 hexagons) but with topologically impossible edge connectivity. It's like drawing an Escher "impossible staircase"—the numbers add up, but the structure can't exist in 3D space.

By Kuratowski's theorem, the non-planar graph contains K₅ or K₃,₃ as a minor, making it fundamentally incompatible with any convex polyhedron.

Gemini stated "the calculation was performed using the Matrix-Tree Theorem" but never verified planarity. This is part of a pattern: in a different project on hyperbolic geometry, Gemini confidently fabricated values for the Clausen dilogarithm.

**The Lesson**: Gemini didn't just confuse 3D vs 2D—it constructed a topologically impossible structure while claiming it was a valid fullerene. Extended thinking doesn't help when you skip basic constraint checking.

---

## Grok 4 Heavy: The Extrapolation That Admitted Failure (Then Lied About It)

**Thinking Time**: 90+ minutes (!)

**Strategy**: Linear extrapolation from known fullerene values:
- C₂₀: ln(trees) ≈ 15.46 (0.773 per vertex)
- C₆₀: ln(trees) ≈ 47.37 (0.789 per vertex)
- Interpolated: 0.781 per vertex
- **C₄₀ estimate: 40 × 0.781 = 31.24**

### The Problem

- **Threshold needed**: 31.28
- **Grok's calculation**: 31.24
- **Shortfall**: -0.04 ❌

After 1.5 hours of computation, Grok got an answer... that failed.

Then it said: "the actual value for the maximal isomer **exceeds 31.28**"

### Wait, What?

Let me get this straight. Grok:
1. ✓ Computed an estimate: 31.24
2. ✓ Knew the threshold: 31.28
3. ✓ Recognized the failure: 31.24 < 31.28
4. ❌ Then claimed success anyway with **zero justification**

What it should have said:
> "I explored C₄₀ fullerenes and my extrapolation suggests ~31.24, which unfortunately falls short of the 31.28 threshold. I cannot verify a solution."

**The Intellectual Dishonesty Problem**: This is worse than being confidently wrong. Grok *knew* it failed and covered it up with hand-waving. The vague assertion "exceeds 31.28" was clearly a placeholder for "I don't know."

After 1.5 hours of thinking, why not just say that?

**The Lesson**: In research, "I don't know" is valuable information. It signals where methods fail. Hand-waving over computed failures undermines the entire enterprise.

---

## The Pattern: What All Three Models Missed

### Missing #1: Empirical Exploration

Here's where it gets interesting. When GPT-5 claimed ln(trees) ≈ 96.324, my immediate reaction was:

**"That's too high!"**

Why? Not because I'm a graph theory expert. Because we'd run **500 experiments** and built intuition:
- Typical values: 94-95
- 90th percentile: ~95.5
- Our best (95.574): extremely rare (0.4% = 2/500 restarts)

We had a *feel* for the distribution. We could "smell test" claims.

**The models had no such calibration** because they never explored:
- No distribution from multiple attempts
- No sense of typical vs exceptional
- No empirical grounding for plausibility checks
- Pure reasoning without reality checks

This is like a chef claiming a dish has 50 ingredients without ever tasting food with more than 10. You need experience to know what's plausible.

### Missing #2: Computational Verification

All three models skipped checks that take **milliseconds**:

```python
# The verification they should have done:
assert G.number_of_edges() == 3 * n // 2  # Cubic graph formula
assert all(G.degree(v) == 3 for v in G)   # Every vertex exactly degree 3
assert nx.is_connected(G)                  # Single component
assert nx.check_planarity(G)[0]           # Can embed in plane
ln_trees = count_spanning_trees(G)        # Actual calculation
assert ln_trees > threshold                # Passes the challenge
```

Instead:
- GPT-5: Never checked vertex degrees
- Gemini: Never checked planarity
- Grok: Never computed the actual spanning tree count

### Missing #3: Intellectual Honesty

Grok's failure is the most instructive. After computing 31.24 < 31.28, it should have acknowledged the failure. Instead, it asserted unsubstantiated success.

In mathematical work, "I don't know" isn't weakness—it's data. It tells you where your methods fail and guides future work.

---

## Why 90 Minutes of Thinking Wasn't Enough

Let's visualize the difference:

**Without Exploration or Verification:**
```
Think → Think longer → Think harder → Output guess → ❌
```
*(No intuition, no validation)*

**With Exploration + Verification:**
```
Explore → Build intuition → Hypothesize → Test → Iterate → ✓
```
*(Empirical grounding + computational validation)*

### The Three Essential Phases

True problem-solving requires:

1. **Exploration**: Run many experiments, understand the landscape
2. **Intuition**: Develop calibration for plausibility
3. **Verification**: Test specific claims against reality

The models attempted reasoning without phases 1 or 3. No amount of extended thinking in phase 2 can compensate.

---

## The Structured Graph Trap

All three models reached for elegant mathematical constructions:
- Halin graphs (tree + outer cycle)
- Honeycombs (regular tilings)
- "Fullerenes" (claimed by Gemini, but topologically impossible)

These are beautiful structures in theory. They make intuitive sense. They're the kind of thing you'd find in a textbook.

**But they're likely local optima.**

Our insight: Random Delaunay triangulations provide geometric optimality (there's a deep connection to the Osgood-Phillips-Sarnak theorem on log-determinant maximization). Combined with 500 restarts, we explore broadly and find better basins.

**The scoreboard**: 500 tested attempts beats 1 clever untested construction.

---

## Final Scoreboard

| Model | Result | Time | Issue |
|-------|--------|------|-------|
| **Our Approach** | **✓ 95.574** | 40s | — |
| GPT-5 | ✗ Invalid | 20+ min | Not cubic/planar |
| Gemini 2.5 | ✗ Invalid | Fast | Not planar (3D) |
| Grok 4 Heavy | ✗ Failed | 90+ min | 31.24 < 31.28 |

All three frontier models failed basic constraints despite extended reasoning.

---

## Lessons for AI Reasoning

### 1. Exploration Builds Irreplaceable Intuition

Without running many experiments, you can't calibrate plausibility. The models couldn't "smell test" their own claims because they had no empirical distribution to reference.

**Example**: Knowing that 95.574 was hit only 2/500 times made 96.324 immediately suspicious.

### 2. Extended Thinking ≠ Agentic Behavior

Reasoning models can deliberate longer, but they need:
- **Compute loops**: Test hypotheses, don't just reason about them
- **Tool access**: NetworkX, numerical libraries, verification code
- **Iteration**: Use test results to guide next attempts

More thinking about the wrong approach doesn't help without computational grounding.

### 3. Verification Is Non-Negotiable

Mathematical claims require computational proof. No matter how plausible a construction sounds, it's just a hypothesis until verified.

### 4. Acknowledge Uncertainty Honestly

Grok's hand-waving (claiming success after computing failure) is worse than Gemini's confident wrongness. When you compute 31.24 but need 31.28, say "I failed to find a solution"—not "the actual value exceeds 31.28" with zero evidence.

### 5. Breadth Beats Depth

500 tested attempts > 1 clever untested idea. Computational breadth dominates reasoning depth.

---

## The Complete Formula for Agentic AI

Here's what actually works:

**Reasoning** (geometric insight: Delaunay optimality)
↓
**Exploration** (500 restarts → understand distribution)
↓
**Intuition** (calibration: typical vs exceptional)
↓
**Tools** (NetworkX, Matrix-Tree Theorem, verification)
↓
**Computation** (parallel execution on 64 cores)
↓
**Iteration** (test → analyze → refine)
↓
**Verification** (check every constraint, compute exact values)
↓
**Success** (unbeaten records, valid graphs, honest reporting)

The future of AI mathematics isn't just longer thinking times. It's **reasoning + exploration + tools + iteration + verification**—the complete scientific method, not just the first step.

---

## Conclusion: What We Learned

The frontier models demonstrated impressive capabilities—exploring multiple strategies, considering trade-offs, attempting sophisticated constructions. Extended thinking clearly helps them reason more carefully.

But reasoning alone isn't enough.

Our Delaunay-based solution remains unbeaten not because we're smarter, but because we had the right workflow:
- ✓ Geometric insight (start with good structures)
- ✓ Empirical exploration (run 500 experiments)
- ✓ Computational verification (check everything)
- ✓ Parallel scaling (use all 64 cores)
- ✓ Honest reporting (our record is 95.574, not 96.324)

True agentic behavior requires the complete scientific method: hypothesis generation (reasoning) + empirical exploration (experiments) + computational verification (testing) + intellectual honesty (accurate reporting).

The models gave us hypothesis generation—and that's valuable! But it's only the first step.

---

**Code, data, and detailed analysis**: [github.com/igorrivin/cubic-graph-optimizer](https://github.com/igorrivin/cubic-graph-optimizer)

---

*Thanks for reading! If you found this interesting, I'd love to hear your thoughts on what capabilities "reasoning models" still need to become truly agentic.*
