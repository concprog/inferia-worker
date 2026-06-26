package dockerclient

import (
	"strings"
	"testing"

	"github.com/inferia/inferia-worker/internal/runtime/recipes"
)

func basePlan() recipes.Plan {
	return recipes.Plan{
		Image:         "docker.io/vllm/vllm-openai:v0.16.0",
		ContainerName: "inferia-vllm-dep-1",
		Cmd:           []string{"meta-llama/Llama-3.1-8B", "--host", "0.0.0.0"},
		Env:           map[string]string{"CUDA_MODULE_LOADING": "LAZY"},
		ContainerPort: 8000,
		HostPort:      19000,
		GPUIndices:    []int{0},
		ReadyPath:     "/health",
	}
}

func TestBuildContainerSpec_HappyPath(t *testing.T) {
	spec, err := BuildContainerSpec(basePlan(), "inferia-models")
	if err != nil {
		t.Fatalf("%v", err)
	}
	if spec.Image != "docker.io/vllm/vllm-openai:v0.16.0" {
		t.Errorf("Image: %q", spec.Image)
	}
	if spec.Name != "inferia-vllm-dep-1" {
		t.Errorf("Name: %q", spec.Name)
	}
	if len(spec.Cmd) == 0 || spec.Cmd[0] != "meta-llama/Llama-3.1-8B" {
		t.Errorf("Cmd: %v", spec.Cmd)
	}
	if v := spec.Env["CUDA_MODULE_LOADING"]; v != "LAZY" {
		t.Errorf("Env: %v", spec.Env)
	}
}

func TestBuildContainerSpec_PortBinding(t *testing.T) {
	spec, _ := BuildContainerSpec(basePlan(), "inferia-models")
	// Expect host binding 127.0.0.1:19000 → 8000/tcp.
	if spec.PortBinding.HostIP != "127.0.0.1" {
		t.Errorf("HostIP: %q", spec.PortBinding.HostIP)
	}
	if spec.PortBinding.HostPort != "19000" {
		t.Errorf("HostPort: %q", spec.PortBinding.HostPort)
	}
	if spec.PortBinding.ContainerPort != "8000/tcp" {
		t.Errorf("ContainerPort: %q", spec.PortBinding.ContainerPort)
	}
}

func TestBuildContainerSpec_GPUDeviceRequest(t *testing.T) {
	p := basePlan()
	p.GPUIndices = []int{0, 1, 3}
	spec, _ := BuildContainerSpec(p, "inferia-models")
	if len(spec.GPUDeviceIDs) != 3 {
		t.Fatalf("expected 3 GPU IDs, got %d", len(spec.GPUDeviceIDs))
	}
	want := []string{"nvidia.com/gpu=0", "nvidia.com/gpu=1", "nvidia.com/gpu=3"}
	for i, id := range want {
		if spec.GPUDeviceIDs[i] != id {
			t.Errorf("GPUDeviceIDs[%d]: %q want %q", i, spec.GPUDeviceIDs[i], id)
		}
	}
}

func TestBuildContainerSpec_NetworkAttached(t *testing.T) {
	spec, _ := BuildContainerSpec(basePlan(), "inferia-models")
	if spec.NetworkName != "inferia-models" {
		t.Errorf("NetworkName: %q", spec.NetworkName)
	}
}

func TestBuildContainerSpec_NoShellInjectionInImage(t *testing.T) {
	// Image must not be interpolated into a shell. We pass it as a struct field;
	// this test guards against future regressions where someone joins it into a
	// shell-out cmd. With our Build, Image is a literal field.
	p := basePlan()
	p.Image = "vllm/vllm-openai;rm -rf /:latest"
	spec, _ := BuildContainerSpec(p, "n")
	if !strings.Contains(spec.Image, ";rm -rf") {
		// We do NOT modify the image string; we expect the caller to have
		// validated upstream. The assertion here is that the spec preserves
		// the bytes verbatim (so downstream sanitisers see them).
		t.Errorf("image should pass through verbatim, got %q", spec.Image)
	}
}

func TestBuildContainerSpec_ContainerNameRequired(t *testing.T) {
	p := basePlan()
	p.ContainerName = ""
	if _, err := BuildContainerSpec(p, "n"); err == nil {
		t.Errorf("expected error for empty ContainerName")
	}
}

func TestBuildContainerSpec_ImageRequired(t *testing.T) {
	p := basePlan()
	p.Image = ""
	if _, err := BuildContainerSpec(p, "n"); err == nil {
		t.Errorf("expected error for empty Image")
	}
}

func TestBuildContainerSpec_ContainerPortRequired(t *testing.T) {
	p := basePlan()
	p.ContainerPort = 0
	if _, err := BuildContainerSpec(p, "n"); err == nil {
		t.Errorf("expected error for ContainerPort=0")
	}
}

func TestBuildContainerSpec_HostPortRequired(t *testing.T) {
	p := basePlan()
	p.HostPort = 0
	if _, err := BuildContainerSpec(p, "n"); err == nil {
		t.Errorf("expected error for HostPort=0")
	}
}

func TestBuildContainerSpec_RestartPolicyNo(t *testing.T) {
	// Workers manage container lifecycle themselves; the container should NOT
	// auto-restart, otherwise a failing model loops without the worker being
	// able to mark the deployment failed.
	spec, _ := BuildContainerSpec(basePlan(), "n")
	if spec.RestartPolicy != "no" {
		t.Errorf("RestartPolicy: %q (expected 'no')", spec.RestartPolicy)
	}
}

func TestBuildContainerSpec_LabelsCarryDeploymentID(t *testing.T) {
	spec, _ := BuildContainerSpec(basePlan(), "n")
	if spec.Labels["inferia.deployment_id"] == "" {
		t.Errorf("labels missing inferia.deployment_id: %v", spec.Labels)
	}
	if spec.Labels["inferia.managed_by"] != "inferia-worker" {
		t.Errorf("labels managed_by: %v", spec.Labels)
	}
}

func TestBuildContainerSpec_GPUCapabilitiesIncludeNvidia(t *testing.T) {
	spec, _ := BuildContainerSpec(basePlan(), "n")
	found := false
	for _, group := range spec.GPUCapabilities {
		for _, cap := range group {
			if cap == "gpu" {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("expected 'gpu' capability: %v", spec.GPUCapabilities)
	}
}
