import yaml

# Load the environment.yml file
with open("./configs/environment.yml", "r") as f:
    config = yaml.safe_load(f)

# Choose environment (local or aws)
env = "local"   # or "aws"

raw_data_path = config[env]["raw_data"]
processed_data_path = config[env]["processed_data"]
models_path = config[env]["models"]

print("Raw data path:", raw_data_path)
print("Processed data path:", processed_data_path)
print("Models path:", models_path)
