---
title: Food Image Recognition for Healthy Eating
emoji: 🍽️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

#  Food Image Recognition for Healthy Eating

A deep learning application to classify food images using **EfficientNetB0** and display nutritional information (calories, protein, fat, carbs) to promote healthy eating habits.


##  Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app_streamlit.py

Then open your browser at http://localhost:8501.

Run with Docker
# Build the image
docker build -t food-recognition-app .

# Run the container
docker run -p 8501:8501 food-recognition-app

Then open http://localhost:8501 in your browser.

Run Unit Tests
python -m unittest discover tests/
Tests cover:

Model input/output shape validation.

File existence errors (FileNotFoundError).

Invalid file extensions (.txt, etc.) → ValueError.

Corrupted/empty images → ValueError.

 Deployment Strategy
Chosen: Cloud deployment using Docker containers.

CI/CD Pipeline (GitHub Actions)
On every push or pull request, the pipeline automatically:

Installs dependencies.

Runs unit tests (tests/).

Performs a quick sample evaluation (model_evaluation.py --test-sample).

Check the "Actions" tab in the repository for results.




