package recipes

// ollamaRecipe wraps the ollama/ollama image. Ollama serves at 11434 and the
// model is pulled on first request via its native /api/pull endpoint. We
// pre-pull at start time by invoking ollama with a wrapper command.
//
// Important: this recipe runs `ollama serve` only — model pull is requested
// by the worker as a separate HTTP call to /api/pull after the container is
// up. That keeps the entrypoint pristine and lets readiness probe rely on
// ollama's HTTP surface.
type ollamaRecipe struct {
	image     string
	port      int
	readyPath string
}

func (r ollamaRecipe) BuildPlan(in BuildInput) (Plan, error) {
	if err := validate(in); err != nil {
		return Plan{}, err
	}
	model := stripScheme(in.ArtifactURI)
	// Pull-after-ready is handled by the runtime layer; we still annotate the
	// container with the model name via env so logs are searchable.
	env := mergeEnv(in.Env, map[string]string{
		"OLLAMA_HOST":          "0.0.0.0:11434",
		"INFERIA_OLLAMA_MODEL": model,
	})
	return Plan{
		Image:         r.image,
		ContainerName: containerName("inferia-ollama", in.DeploymentID),
		Cmd:           []string{"serve"},
		Env:           env,
		ContainerPort: r.port,
		HostPort:      in.HostPort,
		GPUIndices:    in.GPUIndices,
		ReadyPath:     r.readyPath,
	}, nil
}

// infinityRecipe runs michaelf34/infinity, an OpenAI-compatible embedding
// server.
type infinityRecipe struct {
	image     string
	port      int
	readyPath string
}

func (r infinityRecipe) BuildPlan(in BuildInput) (Plan, error) {
	if err := validate(in); err != nil {
		return Plan{}, err
	}
	model := stripScheme(in.ArtifactURI)
	cfg := sanitiseConfig(in.Config)

	cmd := []string{
		"v2",
		"--model-id", model,
		"--port", "7997",
		"--host", "0.0.0.0",
	}
	if v, ok := cfg["max_batch_size"]; ok {
		cmd = append(cmd, "--batch-size", cliArg(v))
	}
	if v, ok := cfg["dtype"]; ok {
		cmd = append(cmd, "--dtype", cliArg(v))
	}
	env := mergeEnv(in.Env, nil)
	return Plan{
		Image:         r.image,
		ContainerName: containerName("inferia-infinity", in.DeploymentID),
		Cmd:           cmd,
		Env:           env,
		ContainerPort: r.port,
		HostPort:      in.HostPort,
		GPUIndices:    in.GPUIndices,
		ReadyPath:     r.readyPath,
	}, nil
}

// tritonRecipe runs NVIDIA Triton Inference Server. Triton expects a model
// repository mounted as a volume; the runtime layer handles that mount based
// on the artifact URI. The cmd here only specifies the model-repository
// location inside the container.
type tritonRecipe struct {
	image     string
	port      int
	readyPath string
}

func (r tritonRecipe) BuildPlan(in BuildInput) (Plan, error) {
	if err := validate(in); err != nil {
		return Plan{}, err
	}
	// Triton (NVIDIA Inference Server) is GPU-only.
	if err := requireGPU(in); err != nil {
		return Plan{}, err
	}
	env := mergeEnv(in.Env, nil)
	return Plan{
		Image:         r.image,
		ContainerName: containerName("inferia-triton", in.DeploymentID),
		Cmd: []string{
			"tritonserver",
			"--model-repository=/models",
			"--allow-http=true",
			"--http-port=8000",
		},
		Env:           env,
		ContainerPort: r.port,
		HostPort:      in.HostPort,
		GPUIndices:    in.GPUIndices,
		ReadyPath:     r.readyPath,
	}, nil
}

// diffusionRecipe runs inferia-diffusion for image/video generation. The image
// id (model checkpoint) is passed via env so the entrypoint stays uniform.
type diffusionRecipe struct {
	image     string
	port      int
	readyPath string
}

func (r diffusionRecipe) BuildPlan(in BuildInput) (Plan, error) {
	if err := validate(in); err != nil {
		return Plan{}, err
	}
	// inferia-diffusion runs image/video generation pipelines that require GPU.
	if err := requireGPU(in); err != nil {
		return Plan{}, err
	}
	env := mergeEnv(in.Env, map[string]string{
		"DIFFUSION_MODEL": stripScheme(in.ArtifactURI),
		"DIFFUSION_PORT":  "8000",
	})
	return Plan{
		Image:         r.image,
		ContainerName: containerName("inferia-diff", in.DeploymentID),
		Cmd:           nil, // use image's default entrypoint
		Env:           env,
		ContainerPort: r.port,
		HostPort:      in.HostPort,
		GPUIndices:    in.GPUIndices,
		ReadyPath:     r.readyPath,
	}, nil
}
