package shellbridge

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
)

// ResolveContainer picks the container to operate on. Mirrors the
// precedence chain in internal/admin so /v1/shell, /v1/logs, and the
// channel-tunneled paths all behave the same:
//
//  1. explicit containerID (operator override)
//  2. deploymentID through the runtime registry
//  3. deploymentID via docker ps (recovers from worker restart that
//     drops the in-memory registry while the model container is still up)
//  4. the runtime's first loaded deployment when both are empty
//  5. the most-recent inferia-managed container (last-ditch fallback)
//
// Returns an error with the explanation string when nothing resolves.
func ResolveContainer(cli *client.Client, rt Runtime, containerID, deploymentID string) (string, error) {
	if containerID != "" {
		return containerID, nil
	}
	if deploymentID != "" {
		if rt != nil {
			if cid := rt.ContainerForDeployment(deploymentID); cid != "" {
				return cid, nil
			}
		}
		if cid := lookupByDeploymentID(cli, deploymentID); cid != "" {
			return cid, nil
		}
		return "", fmt.Errorf("deployment %q has no running container", deploymentID)
	}
	if rt != nil {
		if loaded := rt.LoadedDeployments(); len(loaded) > 0 {
			if cid := rt.ContainerForDeployment(loaded[0]); cid != "" {
				return cid, nil
			}
		}
	}
	if cid := lookupMostRecent(cli); cid != "" {
		return cid, nil
	}
	return "", errors.New("no active deployment on this worker")
}

func lookupByDeploymentID(cli *client.Client, depID string) string {
	if cli == nil || depID == "" {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	list, err := cli.ContainerList(ctx, container.ListOptions{
		All:     false,
		Filters: filters.NewArgs(filters.KeyValuePair{Key: "name", Value: depID}),
	})
	if err != nil || len(list) == 0 {
		return ""
	}
	for _, c := range list {
		for _, n := range c.Names {
			if strings.Contains(n, depID) {
				return c.ID
			}
		}
	}
	return list[0].ID
}

func lookupMostRecent(cli *client.Client) string {
	if cli == nil {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	list, err := cli.ContainerList(ctx, container.ListOptions{
		Filters: filters.NewArgs(filters.KeyValuePair{Key: "name", Value: "inferia-"}),
	})
	if err != nil || len(list) == 0 {
		return ""
	}
	return list[0].ID
}
