from mlflow import MlflowClient

c = MlflowClient("http://localhost:5000")
versions = c.search_model_versions("name='iris-random-forest'")
for v in versions:
    print(f"version={v.version}  aliases={v.aliases}  status={v.status}")

print()
for alias in ["champion", "challenger"]:
    try:
        mv = c.get_model_version_by_alias("iris-random-forest", alias)
        print(f"alias '{alias}' → version {mv.version}")
    except Exception as e:
        print(f"alias '{alias}' → not set ({e})")
