package telemetry

import (
	"os"
	"regexp"
	"strconv"
	"strings"
)

// sectorSize is the conventional /proc/diskstats sector size in bytes (always
// 512 regardless of the device's physical/logical block size — a kernel ABI
// constant).
const sectorSize = 512

// DiskCounters holds cumulative host disk byte counters (summed over physical
// block devices only).
type DiskCounters struct {
	ReadBytes  uint64
	WriteBytes uint64
}

func readProcDiskStats() (string, error) {
	b, err := os.ReadFile("/proc/diskstats")
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ReadDisk returns cumulative read/write byte counters across physical block
// devices. Rate derivation is the caller's job.
func ReadDisk() (DiskCounters, error) {
	return readDiskFrom(readProcDiskStats)
}

func readDiskFrom(r reader) (DiskCounters, error) {
	s, err := r()
	if err != nil {
		return DiskCounters{}, err
	}
	rd, wr, err := parseDiskStats(s)
	if err != nil {
		return DiskCounters{}, err
	}
	return DiskCounters{ReadBytes: rd, WriteBytes: wr}, nil
}

// physicalDiskRe matches whole-disk device names (sda, sdb, xvda, vda,
// nvme0n1) but NOT partitions (sda1, nvme0n1p1). loop*/ram*/dm-* are rejected
// by not matching the prefix set.
var physicalDiskRe = regexp.MustCompile(`^(sd[a-z]+|xvd[a-z]+|vd[a-z]+|nvme[0-9]+n[0-9]+)$`)

func isPhysicalDisk(name string) bool {
	return physicalDiskRe.MatchString(name)
}

// parseDiskStats sums sectorsRead (field index 5) and sectorsWritten (field
// index 9) across physical devices, scaled to bytes.
// /proc/diskstats tokens: [major, minor, name, reads, readsMerged,
//   sectorsRead, readTicks, writes, writesMerged, sectorsWritten, ...]
func parseDiskStats(s string) (read, write uint64, err error) {
	for _, line := range strings.Split(s, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 10 {
			continue
		}
		name := fields[2]
		if !isPhysicalDisk(name) {
			continue
		}
		sectorsRead, e1 := strconv.ParseUint(fields[5], 10, 64)
		sectorsWritten, e2 := strconv.ParseUint(fields[9], 10, 64)
		if e1 != nil || e2 != nil {
			continue
		}
		read += sectorsRead * sectorSize
		write += sectorsWritten * sectorSize
	}
	return read, write, nil
}
