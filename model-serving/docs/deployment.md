# Deployment

The repository includes a single-service Docker image, a Compose monitoring
stack, and a Kubernetes development manifest. Start with the Docker image and
the generated example artifact before adapting the larger examples.

## Docker

```bash
docker build --tag highway-rl-serving:local .
docker run --rm --publish 8080:8080 highway-rl-serving:local
```

The image runs as a non-root user, listens on port 8080, and generates the test
artifact during the build. Check it with:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metadata
```

To serve a real artifact, mount a read-only version directory and set
`ARTIFACT_DIR` and `MODEL_VERSION` explicitly.

## Compose

`docker-compose.yml` adds Prometheus and Grafana around the service. Review the
published ports, volumes, wildcard CORS and host settings, and pinned monitoring
images before starting it:

```bash
docker compose config
docker compose up --build
```

Stop the stack with `docker compose down`. Add `--volumes` only when you intend
to delete the local monitoring data.

## Kubernetes

`deploy/k8s/deployment.yaml` is a local example. It expects an image named
`model-serving:v0.1.0` already loaded on the cluster because its pull policy is
`Never`.

Render and inspect the manifest first:

```bash
kubectl apply --dry-run=client -f deploy/k8s/deployment.yaml
kubectl diff -f deploy/k8s/deployment.yaml
```

Before using it outside a disposable cluster, pin an immutable image digest,
restrict network exposure, set resource requests and limits from measurements,
configure CORS and trusted hosts, provide a durable artifact source, and add the
environment's policy and observability requirements.
