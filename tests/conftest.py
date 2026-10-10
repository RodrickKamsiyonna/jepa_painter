import torch


def pytest_sessionstart(session):
    # Small 32-pixel tests are faster without large CPU thread pools.
    torch.set_num_threads(2)
