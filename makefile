
.ONESHELL:
DATA_DIR = data/full_data
CONFIG_DIR = config
COUNTS_FILE = data/counts678.json
TRAINER_CONFIG = config/lanl_config_trainer.json
FLAGS = -tc $(TRAINER_CONFIG) -df $(DATA_DIR)

TRAIN_MODEL = python log_analyzer/train_model.py


ifdef BIDIRECTIONAL
FLAGS += --bidir
endif

ifdef CUDA
FLAGS += --use-cuda
endif

ifdef WANDB
FLAGS += --wandb-sync
endif

ifdef LOAD
FLAGS += --load-model $(LOAD)
endif

ifdef PYFLAGS
FLAGS += $(PYFLAGS)
endif

tox:
	tox .

lint:
	tox . -e py3-lint

install:
	pip install -e .

profile-%: config/lanl_config_%.json
	python -m cProfile $(TRAIN_MODEL) lstm -cf $(COUNTS_FILE) -mc $^ 

.PHONY: %_word
%_word-field: TOKENIZATION=word-field
%_word-global: TOKENIZATION=word-global
%_char: TOKENIZATION=char

%_word-field %_word-global %_char: config/lanl_config_%.json
	$(TRAIN_MODEL) $* $(TOKENIZATION) -cf $(COUNTS_FILE) -mc $^ $(FLAGS)

