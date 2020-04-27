.PHONY: train_def predict_def run_def train_opt predict_opt run_opt optimize

VENV_NAME = vnlife

# Makefile
venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || virtualenv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install -r requirements.txt
	$(VENV_NAME)/bin/python3 setup.py develop
	ipython kernel install --user --name=$(VENV_NAME)
	touch $(VENV_NAME)/bin/activate

train_rf:
	cd src; ../$(VENV_NAME)/bin/python3 train.py --model_json ../params/def_rf_model.json \
																						   --split_ratio 0.9

train_opt_rf:
	cd src; ../$(VENV_NAME)/bin/python3 train.py --model_json ../params/opt_rf_model.json \
																						   --split_ratio 0.9

train_xgb:
	cd src; ../$(VENV_NAME)/bin/python3 train.py --model_json ../params/def_xgb_model.json \
																						   --split_ratio 0.9

train_opt_xgb:
	cd src; ../$(VENV_NAME)/bin/python3 train.py --model_json ../params/opt_xgb_model.json \
																							 --split_ratio 0.9

optimize_rf:
	cd src; ../$(VENV_NAME)/bin/python3 optimize.py --model_json ../params/def_rf_model.json

optimize_xgb:
	cd src; ../$(VENV_NAME)/bin/python3 optimize.py --model_json ../params/def_xgb_model.json

features:
	cd src; ../$(VENV_NAME)/bin/python3 features.py

clv:
	cd src; ../$(VENV_NAME)/bin/python3 clv.py
