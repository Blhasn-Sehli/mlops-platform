from mlflow import MlflowClient

c = MlflowClient("http://localhost:5000")

# champion = most recent version (set by train.py)
# challenger = previous version (for comparison)
versions = sorted(
    c.search_model_versions("name='iris-random-forest'"),
    key=lambda v: int(v.version),
)
if len(versions) < 2:
    print(f"Only {len(versions)} version(s) available — need at least 2 for A/B testing")
    print("Run training twice to generate a second version, then re-run this script")
else:
    champion_v = versions[-1].version
    challenger_v = versions[-2].version
    c.set_registered_model_alias("iris-random-forest", "champion", champion_v)
    c.set_registered_model_alias("iris-random-forest", "challenger", challenger_v)
    print(f"champion  → version {champion_v}")
    print(f"challenger → version {challenger_v}")
