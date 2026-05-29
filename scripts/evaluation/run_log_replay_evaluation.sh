export PY123D_DATA_ROOT=YOUR DATA
export NAV123D_EXP_ROOT=YOUR EXP ROOT

python src/nav123d/script/run_evaluation.py \
    test_scene_filter=nuplan-mini_test \
    agent=log_replay_agent \
    experiment_name=testing_nuplan-mini_test_eval \
    hydra/job_logging=default \
    hydra/hydra_logging=colorlog