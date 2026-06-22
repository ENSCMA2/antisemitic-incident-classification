from datasets import load_dataset

TG_data = load_dataset("skg/toxigen-data", name="train", use_auth_token="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") # 250k training examples
TG_annotations = load_dataset("skg/toxigen-data", name="annotated", use_auth_token="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") # Human study

TG_data.to_csv("tg_train.csv")
TG_annotations.to_csv("tg_ann.csv")