import cloudpickle


def check():
    path = "/home/opc/AlgoMLB/.data/models/pa_outcome_v1.1.pkl"
    try:
        with open(path, "rb") as f:
            m = cloudpickle.load(f)
        print("Type:", type(m))
        actual = m.model if hasattr(m, "model") else m
        print("Actual Type:", type(actual))

        if hasattr(actual, "feature_names_in_"):
            print("Has feature_names_in_: Yes")
        else:
            print("Has feature_names_in_: No")

        if hasattr(actual, "get_booster"):
            print("Booster features:", actual.get_booster().feature_names[:3])

    except Exception as e:
        print("Error:", e)


check()
