# 🚀 DockerSklearn API

**Containerized scikit-learn ML model served via Flask API.**  
Perfect demo of MLOps basics: training → containerization → inference.

[![Docker Build](https://img.shields.io/badge/Docker-Build-blue?logo=docker)](https://hub.docker.com)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)

## 🎯 Features

- Train simple logistic regression classifier on synthetic data
- `/predict` endpoint for real-time inference
- `/health` + `/model-info` for monitoring
- Fully Dockerized with Compose support
- Health checks and production‑ready Dockerfile

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose

### Run Locally

```bash
# Clone & run
git clone <your-repo> dockerscikit-learn-api
cd dockerscikit-learn-api
docker compose up --build
