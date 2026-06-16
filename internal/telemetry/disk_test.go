package telemetry

import "testing"

// /proc/diskstats columns: major minor name reads readsMerged sectorsRead(f6)
// readTicks writes writesMerged sectorsWritten(f10) ...
const sampleDiskStats = `   8       0 sda 100 0 200 0 50 0 400 0 0 0 0
   8       1 sda1 10 0 20 0 5 0 40 0 0 0 0
 259       0 nvme0n1 1000 0 2000 0 500 0 4000 0 0 0 0
 259       1 nvme0n1p1 1 0 2 0 1 0 4 0 0 0 0
   7       0 loop0 1 0 9999 0 1 0 9999 0 0 0 0
   1       0 ram0 1 0 8888 0 1 0 8888 0 0 0 0
`

func TestParseDiskStats_PhysicalOnly_ScaledBy512(t *testing.T) {
	rd, wr, err := parseDiskStats(sampleDiskStats)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	wantRead := uint64((200 + 2000) * 512)
	wantWrite := uint64((400 + 4000) * 512)
	if rd != wantRead || wr != wantWrite {
		t.Fatalf("rd=%d wr=%d, want rd=%d wr=%d", rd, wr, wantRead, wantWrite)
	}
}

func TestParseDiskStats_Empty(t *testing.T) {
	rd, wr, err := parseDiskStats("")
	if err != nil || rd != 0 || wr != 0 {
		t.Fatalf("empty: rd=%d wr=%d err=%v", rd, wr, err)
	}
}

func TestParseDiskStats_MalformedSkipped(t *testing.T) {
	in := "short line\n   8 0 sda 1 2\n   8 0 sda 1 0 100 0 0 0 200 0 0 0 0\n"
	rd, wr, err := parseDiskStats(in)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if rd != 100*512 || wr != 200*512 {
		t.Fatalf("rd=%d wr=%d, want %d/%d", rd, wr, 100*512, 200*512)
	}
}

func TestIsPhysicalDisk(t *testing.T) {
	cases := map[string]bool{
		"sda": true, "sdb": true, "xvda": true, "nvme0n1": true, "vda": true,
		"sda1": false, "nvme0n1p1": false, "loop0": false, "ram0": false,
		"dm-0": false, "": false,
	}
	for name, want := range cases {
		if got := isPhysicalDisk(name); got != want {
			t.Errorf("isPhysicalDisk(%q)=%v, want %v", name, got, want)
		}
	}
}
