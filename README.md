# SQM Assignment: Quatromino Tiling with Minimum Cost

## Problem Description

You are given a rectangular game board **S** of dimensions:

* ( 3 \leq a, b \leq 20 )
* Total number of cells ( ab \geq 15 )
* Each cell contains a natural number from the interval ([1, 100])

Let **P = {T, Z}** be the set of allowed quatromino pieces.

Although standard quatromino shapes include **I, L, O, T, Z**, in this task we work **only with shapes T and Z**.

---

## Definitions

### Board Covering

A covering of the board **S** using quatrominoes from set **P** is defined as:

* A placement of non-overlapping quatrominoes,
* Pieces may be rotated and flipped,
* No additional quatromino can be placed afterwards,
* The number of used **T** and **Z** pieces must be equal or differ by at most 1.

Some cells may remain uncovered.

---

### Cost of a Covering

The **cost** of a covering is defined as:

> The sum of the values of all uncovered cells.

---

## Example

For a 4×4 board, suppose uncovered cells contain values:

```
5, 7, 1, 4
```

Then the cost of the covering is:

```
5 + 7 + 1 + 4 = 17
```

---

## Task

Find a covering of board **S** using quatrominoes from **P = {T, Z}** such that the covering cost is **minimal**.

---

## Output Format

The output must be a matrix representation of the final covering.

* Each placed quatromino is represented by a label:

  ```
  [T|Z]<index>
  ```

  where:

  * `T` or `Z` is the piece type,
  * `<index>` is a unique sequential number for that piece.

* Uncovered cells must contain their original numeric value.

### Example Output (Not Optimal)

```
 T1  T1  T1   5
 Z1  T1   7   Z2
 Z1  Z1  Z2   Z2
  1  Z1  Z2    4
```

You may optionally implement a graphical visualization of the solution.

---

# Sequential Algorithm

The solution should use:

> **Branch and Bound – Depth First Search (BB-DFS)**

---

## Lower Bounds

### Trivial Lower Bound

Let:

```
k = ab mod 4
```

The trivial lower bound on the covering cost is:

> The sum of the k smallest cell values.

If ( ab ) is divisible by 4, then the trivial lower bound is:

```
0
```

### Non-Trivial Lower Bound

No non-trivial lower bound is known.

---

# Implementation Notes

## Note 1 – Systematic Board Filling

At the beginning, all cells are undecided.

Do NOT place pieces randomly.

Instead:

1. Always select the nearest undecided cell (e.g., scanning from the top-left corner).
2. Either:

   * Place a valid quatromino covering that cell, or
   * Mark the cell as uncovered.

For efficiency, prefer decisions that reduce the covering cost.

---

## Note 2 – Pruning

You must prune non-promising branches:

* If the current cost (sum of uncovered cells so far) is **greater than or equal to** the best known minimum.
* If it is impossible to satisfy the parity constraint between the number of T and Z pieces.

---

## Note 3 – Early Termination

If a covering is found whose cost equals the trivial lower bound:

> The computation can terminate immediately.

---

# Parallel Algorithm

The parallel version should follow a:

> **Master–Slave architecture**

The Master distributes subproblems to Slave processes, which independently explore branches of the state space using the Branch and Bound approach.

---

# Summary

Your goal is to:

* Systematically explore the state space,
* Use Branch and Bound pruning,
* Respect parity constraints between T and Z pieces,
* Minimize the sum of uncovered cell values,
* Output a matrix-form solution with uniquely labeled pieces.
