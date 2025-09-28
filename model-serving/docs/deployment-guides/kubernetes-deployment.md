# Kubernetes Deployment Guide

This guide covers deploying the CarlaRL Policy-as-a-Service on Kubernetes clusters.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Monitoring Setup](#monitoring-setup)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)
- [Security](#security)

## Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Docker registry access
- Helm 3.0+ (optional)
- At least 2 nodes with 4GB RAM each

## Quick Start

### 1. Build and Push Image

```bash
# Build image
docker build -t your-registry/carla-rl-serving:latest .

# Push to registry
docker push your-registry/carla-rl-serving:latest
```

### 2. Deploy to Kubernetes

```bash
# Apply basic deployment
kubectl apply -f deploy/k8s/deployment.yaml

# Check deployment status
kubectl get pods -l app=model-serving

# Check service
kubectl get svc model-serving-service
```

### 3. Test Deployment

```bash
# Port forward for testing
kubectl port-forward svc/model-serving-service 8080:80

# Test health endpoint
curl http://localhost:8080/healthz

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [{
      "speed": 25.5,
      "steering": 0.1,
      "sensors": [0.8, 0.2, 0.5, 0.9, 0.1]
    }],
    "deterministic": true
  }'
```

## Production Deployment

### 1. Namespace Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: carla-rl-serving
  labels:
    name: carla-rl-serving
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: carla-rl-serving-quota
  namespace: carla-rl-serving
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    persistentvolumeclaims: "4"
```

### 2. ConfigMap for Artifacts

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-serving-artifacts
  namespace: carla-rl-serving
data:
  model_card.yaml: |
    model_name: "carla-ppo"
    version: "v0.1.0"
    model_type: "pytorch"
    description: "CarlaRL PPO policy for autonomous driving"
    input_shape: [5]
    output_shape: [3]
    framework_version: "2.1.0"
    performance_metrics:
      reward: 850.5
      success_rate: 0.95
    artifact_hashes:
      model.pt: "sha256:abc123..."
      preprocessor.pkl: "sha256:def456..."
```

### 3. Production Deployment

```yaml
# k8s/production-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  namespace: carla-rl-serving
  labels:
    app: model-serving
    version: v0.1.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: model-serving
  template:
    metadata:
      labels:
        app: model-serving
        version: v0.1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: model-serving
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: model-serving
        image: your-registry/carla-rl-serving:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        env:
        - name: ARTIFACT_DIR
          value: "/app/artifacts"
        - name: MODEL_VERSION
          value: "v0.1.0"
        - name: USE_GPU
          value: "0"
        - name: LOG_LEVEL
          value: "info"
        - name: WORKERS
          value: "2"
        - name: CORS_ORIGINS
          value: "https://yourdomain.com"
        - name: ALLOWED_HOSTS
          value: "yourdomain.com,api.yourdomain.com"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
            ephemeral-storage: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
            ephemeral-storage: 2Gi
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: artifacts
          mountPath: /app/artifacts
          readOnly: true
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: artifacts
        configMap:
          name: model-serving-artifacts
      - name: config
        configMap:
          name: model-serving-config
      - name: logs
        emptyDir: {}
      nodeSelector:
        kubernetes.io/os: linux
      tolerations:
      - key: "carla-rl-serving"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - model-serving
              topologyKey: kubernetes.io/hostname
      restartPolicy: Always
```

### 4. Service Configuration

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-serving-service
  namespace: carla-rl-serving
  labels:
    app: model-serving
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
spec:
  selector:
    app: model-serving
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 8080
    protocol: TCP
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: model-serving-headless
  namespace: carla-rl-serving
  labels:
    app: model-serving
spec:
  selector:
    app: model-serving
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  clusterIP: None
```

### 5. Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-serving-ingress
  namespace: carla-rl-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "30"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: model-serving-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: model-serving-service
            port:
              number: 80
```

## Configuration

### 1. Service Account

```yaml
# k8s/serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: model-serving
  namespace: carla-rl-serving
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: carla-rl-serving
  name: model-serving
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: model-serving
  namespace: carla-rl-serving
subjects:
- kind: ServiceAccount
  name: model-serving
  namespace: carla-rl-serving
roleRef:
  kind: Role
  name: model-serving
  apiGroup: rbac.authorization.k8s.io
```

### 2. ConfigMap for Configuration

```yaml
# k8s/configmap-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-serving-config
  namespace: carla-rl-serving
data:
  config.yaml: |
    server:
      host: "0.0.0.0"
      port: 8080
      workers: 2
      log_level: "info"
    
    model:
      artifact_dir: "/app/artifacts"
      version: "v0.1.0"
      use_gpu: false
    
    monitoring:
      enable_metrics: true
      metrics_port: 8080
      enable_tracing: true
    
    optimization:
      enable_memory_pinning: true
      cache_size: 1000
      batch_size: 1
```

### 3. Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-serving-secrets
  namespace: carla-rl-serving
type: Opaque
data:
  # Base64 encoded values
  api-key: <base64-encoded-api-key>
  database-url: <base64-encoded-database-url>
```

## Monitoring Setup

### 1. Prometheus ServiceMonitor

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: model-serving
  namespace: carla-rl-serving
  labels:
    app: model-serving
spec:
  selector:
    matchLabels:
      app: model-serving
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
```

### 2. Grafana Dashboard

```yaml
# k8s/grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-serving-dashboard
  namespace: carla-rl-serving
  labels:
    grafana_dashboard: "1"
data:
  dashboard.json: |
    {
      "dashboard": {
        "title": "CarlaRL Policy-as-a-Service",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total[5m])",
                "legendFormat": "{{method}} {{endpoint}}"
              }
            ]
          },
          {
            "title": "Latency",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "P50"
              },
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "P95"
              }
            ]
          }
        ]
      }
    }
```

### 3. Alerting Rules

```yaml
# k8s/alerting-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: model-serving-alerts
  namespace: carla-rl-serving
  labels:
    app: model-serving
spec:
  groups:
  - name: model-serving
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.01
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High latency detected"
        description: "P95 latency is above 10ms for 5 minutes"
    
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is above 5% for 5 minutes"
```

## Scaling

### 1. Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
  namespace: carla-rl-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 2. Vertical Pod Autoscaler

```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: model-serving-vpa
  namespace: carla-rl-serving
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: model-serving
      minAllowed:
        cpu: 100m
        memory: 256Mi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
```

### 3. Cluster Autoscaler

```yaml
# k8s/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      containers:
      - name: cluster-autoscaler
        image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/your-cluster-name
        - --balance-similar-node-groups
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
```

## Troubleshooting

### 1. Pod Issues

```bash
# Check pod status
kubectl get pods -n carla-rl-serving

# Check pod logs
kubectl logs -f deployment/model-serving -n carla-rl-serving

# Describe pod for events
kubectl describe pod <pod-name> -n carla-rl-serving

# Check resource usage
kubectl top pods -n carla-rl-serving
```

### 2. Service Issues

```bash
# Check service endpoints
kubectl get endpoints -n carla-rl-serving

# Test service connectivity
kubectl run test-pod --image=busybox --rm -it -- nslookup model-serving-service.carla-rl-serving.svc.cluster.local

# Check service configuration
kubectl get svc -n carla-rl-serving -o yaml
```

### 3. Ingress Issues

```bash
# Check ingress status
kubectl get ingress -n carla-rl-serving

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Test ingress connectivity
curl -H "Host: api.yourdomain.com" http://<ingress-ip>/healthz
```

### 4. Resource Issues

```bash
# Check resource quotas
kubectl describe quota -n carla-rl-serving

# Check node resources
kubectl top nodes

# Check pod resource usage
kubectl top pods -n carla-rl-serving --containers
```

## Security

### 1. Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-serving-netpol
  namespace: carla-rl-serving
spec:
  podSelector:
    matchLabels:
      app: model-serving
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### 2. Pod Security Policy

```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: model-serving-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 3. RBAC Configuration

```yaml
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: model-serving
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: model-serving
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: model-serving
subjects:
- kind: ServiceAccount
  name: model-serving
  namespace: carla-rl-serving
```

## Helm Chart

### 1. Chart Structure

```
charts/carla-rl-serving/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── serviceaccount.yaml
│   └── hpa.yaml
└── README.md
```

### 2. Values.yaml

```yaml
# charts/carla-rl-serving/values.yaml
replicaCount: 3

image:
  repository: your-registry/carla-rl-serving
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: model-serving-tls
      hosts:
        - api.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}
tolerations: []
affinity: {}
```

### 3. Installation

```bash
# Add Helm repository
helm repo add carla-rl https://your-helm-repo.com
helm repo update

# Install chart
helm install carla-rl-serving carla-rl/carla-rl-serving \
  --namespace carla-rl-serving \
  --create-namespace \
  --values values.yaml

# Upgrade chart
helm upgrade carla-rl-serving carla-rl/carla-rl-serving \
  --namespace carla-rl-serving \
  --values values.yaml
```

## Next Steps

- [Bare Metal Deployment Guide](bare-metal-deployment.md)
- [Configuration Reference](../configuration-reference.md)
- [Performance Tuning Guide](../performance-tuning/performance-tuning.md)
- [Monitoring Setup Guide](../monitoring/monitoring-setup.md)
