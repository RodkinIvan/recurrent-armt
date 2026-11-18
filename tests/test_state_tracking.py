import json

import pytest
import torch

from recurrent_armt.state_tracking import GROUPS, WordProblem


@pytest.mark.parametrize("name,size", [("parity", 2), ("z60", 60), ("a5", 60), ("s5", 120)])
def test_group_axioms(name, size):
    t = torch.tensor(GROUPS[name]())
    assert t.shape == (size, size)
    idx = torch.arange(size)
    assert (t[0] == idx).all() and (t[:, 0] == idx).all()                      # identity is 0
    for row in (t, t.T):                                                      # Latin square: inverses exist, cancellation
        assert (row.sort(dim=1).values == idx).all()
    a, b, c = torch.randint(size, (3, 2000))
    assert (t[t[a, b], c] == t[a, t[b, c]]).all()                             # associative


def test_a5_is_not_abelian_and_parity_is_xor():
    a5 = torch.tensor(GROUPS["a5"]())
    assert (a5 != a5.T).any()
    task = WordProblem("parity")
    x, y = task.sample(8, 50)
    assert (y == x.cumsum(1) % 2).all()


def test_products_match_a_fold():
    task = WordProblem("s5")
    x, y = task.sample(4, 30)
    for b in range(4):
        acc = 0
        for t in range(30):
            acc = task.table[acc, x[b, t]].item()
            assert y[b, t].item() == acc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla kernels need CUDA")
@pytest.mark.parametrize("extra", [["--num_mem_tokens", "2", "--segment_size", "4", "--eval_state_reset",
                                    "--curriculum", "5,8,12", "--advance_acc", "0", "--lr_schedule", "constant"],
                                   ["--num_mem_tokens", "0", "--allow_neg_eigval"]])
def test_training_script_runs(tmp_path, extra):
    from train_state_tracking import main
    main(["--task", "a5", "--out_dir", str(tmp_path), "--steps", "6", "--log_every", "3", "--eval_every", "6",
          "--batch_size", "4", "--train_len", "16", "--eval_lens", "8,20", "--eval_size", "6", "--eval_batch_size", "4",
          "--warmup_steps", "2", *extra])
    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [l["step"] for l in lines] == [3, 6, 6]
    assert [l["train_len"] for l in lines if "train_len" in l] == ([5, 8] if "--curriculum" in extra else [16, 16])
    assert 0 <= lines[-1]["eval/acc_L20"] <= 1
    assert ("eval/reset/acc_L20" in lines[-1]) == ("--eval_state_reset" in extra)
    assert (tmp_path / "final" / "config.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla kernels need CUDA")
def test_memory_arm_must_start_longer_than_a_segment(tmp_path):
    from train_state_tracking import main
    with pytest.raises(SystemExit, match="segment_size"):          # length 4 < segment 5: the memory would never be used
        main(["--task", "a5", "--out_dir", str(tmp_path), "--num_mem_tokens", "4", "--segment_size", "5", "--curriculum", "4,8"])
