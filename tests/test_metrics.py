"""Stage-2 sanity test: the local scorer must reproduce the numbers stated in the PS.
Run:  python -m tests.test_metrics"""
from src.metrics import macro_f05


def main():
    # PS example: predicted 3, 2 correct, truth 2 -> P=2/3, R=1 -> 0.714
    truth = {"S1-1": {"S2-47", "S3-812"}}
    pred = {"S1-1": {"S2-47", "S2-193", "S3-812"}}
    assert abs(macro_f05(pred, truth, ["S1-1"]) - 0.7142857) < 1e-4
    # singletons: empty/empty -> 1.0, any match -> 0.0
    assert macro_f05({}, {"S1-2": set()}, ["S1-2"]) == 1.0
    assert macro_f05({"S1-2": {"S2-1"}}, {"S1-2": set()}, ["S1-2"]) == 0.0
    # macro average over entities
    assert abs(macro_f05({"S1-2": {"S2-1"}}, {"S1-2": set(), "S1-3": set()}, ["S1-2", "S1-3"]) - 0.5) < 1e-9
    print("metrics OK")


if __name__ == "__main__":
    main()
