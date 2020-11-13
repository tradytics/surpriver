start: dockerfile docker-compose.yml
	docker-compose up -d --build

stop:
	docker-compose stop

down:
	docker-compose down

ps:
	docker-compose ps

today: start
	docker-compose exec surpriver \
		python detection_engine.py \
		--top_n 25 \
		--min_volume 5000 \
		--data_granularity_minutes 60 \
		--history_to_use 30 \
		--is_load_from_dictionary 0 \
		--data_dictionary_path 'dictionaries/data_dict.npy' \
		--is_save_dictionary 1 \
		--is_test 0 \
		--future_bars 0

history: start
	docker-compose exec surpriver \
		python detection_engine.py \
		--top_n 25 \
		--min_volume 5000 \
		--data_granularity_minutes 60 \
		--history_to_use 14 \
		--is_load_from_dictionary 1 \
		--data_dictionary_path 'dictionaries/data_dict.npy' \
		--is_save_dictionary 0 \
		--is_test 0 \
		--future_bars 0
