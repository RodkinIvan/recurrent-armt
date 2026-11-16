"""Group word problems, the standard state-tracking benchmark: x_1 .. x_L uniform over a finite group G, and at every
position the label is the running product y_t = x_1 x_2 ... x_t. Generated on the fly, at any length.

- ``parity`` (Z2) and ``z60`` (cyclic, abelian): solvable by a linear RNN whose transitions have negative / complex
  eigenvalues, not by one whose eigenvalues are in [0, 1] (GDN-2 by default).
- ``a5`` (60 elements, the smallest non-solvable group: its word problem is NC1-complete) and ``s5`` (120): no
  fixed-depth linear RNN or transformer solves them at arbitrary length; a nonlinear recurrence can.
The ids are 0 .. |G|-1, the identity is 0, and inputs and labels share them.
"""
from __future__ import annotations

import itertools

import torch


def _sign(p: tuple[int, ...]) -> int:
    inversions = sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
    return -1 if inversions % 2 else 1


def _permutations(n: int, even_only: bool) -> list[list[int]]:
    """Multiplication table of S_n (or A_n): table[a][b] = a∘b, (a∘b)(i) = a(b(i)); identity first."""
    perms = [p for p in itertools.permutations(range(n)) if not even_only or _sign(p) == 1]
    index = {p: i for i, p in enumerate(perms)}
    return [[index[tuple(a[b[i]] for i in range(n))] for b in perms] for a in perms]


def _cyclic(n: int) -> list[list[int]]:
    return [[(a + b) % n for b in range(n)] for a in range(n)]


GROUPS = {
    "parity": lambda: _cyclic(2),
    "z60": lambda: _cyclic(60),
    "a5": lambda: _permutations(5, even_only=True),
    "s5": lambda: _permutations(5, even_only=False),
}


class WordProblem:
    def __init__(self, name: str, device: torch.device | str = "cpu"):
        if name not in GROUPS:
            raise ValueError(f"unknown group {name!r}; one of {sorted(GROUPS)}")
        self.name = name
        self.table = torch.tensor(GROUPS[name](), dtype=torch.long, device=device)
        self.size = self.table.shape[0]

    def products(self, x: torch.Tensor) -> torch.Tensor:
        """Running products of x [B, L]: y[:, t] = x[:, 0] ∘ ... ∘ x[:, t]."""
        y = torch.empty_like(x)
        acc = x[:, 0]
        y[:, 0] = acc
        for t in range(1, x.shape[1]):
            acc = self.table[acc, x[:, t]]
            y[:, t] = acc
        return y

    def sample(self, batch_size: int, length: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(self.size, (batch_size, length), device=self.table.device, generator=generator)
        return x, self.products(x)
