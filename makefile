
.ONESHELL:
DATA_DIR = data/full_data
CONFIG_DIR = config
COUNTS_FILE = data/counts678.json
TRAINER_CONFIG = config/lanl_config_trainer.json
FLAGS = --trainer-config $(TRAINER_CONFIG) --data-folder $(DATA_DIR)

ifdef BIDIRECTIONAL
FLAGS += --bidirectional
endif

tox:
	tox .

lint:
	tox . -e py3-lint

install:
	pip install -e .

.PHONY: %_word
%_word-field: TOKENIZATION=word-field
%_word-global: TOKENIZATION=word-global
%_char: TOKENIZATION=char

%_word-field %_char: config/lanl_config_%.json
	python log_analyzer/train_model.py --model-type $* --counts-file $(COUNTS_FILE) --model-config $^ $(FLAGS) --tokenization $(TOKENIZATION)