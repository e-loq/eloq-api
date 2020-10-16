dockerTag := temp 

docker:
	docker login
	docker build -t ${dockerTag} .
	docker run ${dockerTag}

venv:
	python3 -m venv venv/
	source venv/bin/activate
	pip install -r requirements.txt