package cloudenv

import (
	"os"
	"testing"
)

func TestDetect_EnvOverrideSetsKind(t *testing.T) {
	t.Setenv("INFERIA_RUNTIME_ENV", "aws-ec2")
	t.Setenv("INFERIA_INSTANCE_ID", "i-test-1234")
	t.Setenv("INFERIA_REGION", "us-east-1")
	t.Setenv("INFERIA_AZ", "us-east-1a")
	// Force IMDS path off so we don't depend on real network.
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", "http://127.0.0.1:1") // unreachable

	got := detectFresh() // bypasses cache for tests
	if got.Kind != KindAWSEC2 {
		t.Fatalf("Kind = %q, want %q", got.Kind, KindAWSEC2)
	}
	if got.InstanceID != "i-test-1234" {
		t.Errorf("InstanceID = %q", got.InstanceID)
	}
	if got.Region != "us-east-1" {
		t.Errorf("Region = %q", got.Region)
	}
	if got.AvailabilityZone != "us-east-1a" {
		t.Errorf("AZ = %q", got.AvailabilityZone)
	}
	_ = os.Getenv // keep "os" import alive
}

func TestDetect_NoEnvNoIMDSReturnsLocal(t *testing.T) {
	t.Setenv("INFERIA_RUNTIME_ENV", "")
	t.Setenv("INFERIA_CLOUDENV_IMDS_URL", "http://127.0.0.1:1") // unreachable
	got := detectFresh()
	if got.Kind != KindLocal {
		t.Fatalf("Kind = %q, want %q", got.Kind, KindLocal)
	}
}
