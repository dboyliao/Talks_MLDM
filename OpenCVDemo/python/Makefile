.PHONY: clean

clean:
	rm -f *.pyc
	rm -rf train_images model.txt labels.csv

train:
	@if [ ! -d "train_images" ]; then \
		echo 'generate training images...'; \
		./csv_to_images.py; \
	fi
	./train_model.py
