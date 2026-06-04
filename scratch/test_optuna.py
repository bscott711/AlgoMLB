import optuna

try:
    storage_url = "sqlite:///models/optuna_history.db"
    study_summaries = optuna.get_all_study_summaries(storage=storage_url)
    print([s.study_name for s in study_summaries])
except Exception as e:
    print(e)
