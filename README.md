

NUMBA_ENABLE_PROFILING=1 python -m cProfile -o profile.prof src/train.py --num-episodes=10
uv run -m snakeviz profile.prof 