// Package npy reads NumPy .npy files: the reference arrays the real-checkpoint
// tests compare against. It reads format versions 1 to 3, little-endian,
// C-ordered arrays of the dtypes the references use (floats, signed and unsigned
// integers), and refuses everything else by name instead of guessing.
package npy

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Array is one .npy array.
type Array struct {
	// Dtype is the numpy descriptor, e.g. "<f4".
	Dtype string
	// Shape is the array's shape; empty for a scalar.
	Shape []int
	data  []byte
}

var magic = []byte("\x93NUMPY")

var (
	descrRe   = regexp.MustCompile(`'descr':\s*'([^']*)'`)
	fortranRe = regexp.MustCompile(`'fortran_order':\s*(True|False)`)
	shapeRe   = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)
)

// Len is the number of elements.
func (a *Array) Len() int {
	n := 1
	for _, d := range a.Shape {
		n *= d
	}
	return n
}

// Read parses a .npy file.
func Read(path string) (*Array, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	a, err := Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return a, nil
}

// Parse parses the bytes of a .npy file.
func Parse(raw []byte) (*Array, error) {
	if len(raw) < 10 || !bytes.Equal(raw[:6], magic) {
		return nil, fmt.Errorf("not a .npy file (no magic)")
	}
	major := raw[6]
	var hlen, start int
	switch major {
	case 1:
		hlen, start = int(binary.LittleEndian.Uint16(raw[8:10])), 10
	case 2, 3:
		if len(raw) < 12 {
			return nil, fmt.Errorf("truncated version %d header", major)
		}
		hlen, start = int(binary.LittleEndian.Uint32(raw[8:12])), 12
	default:
		return nil, fmt.Errorf("unsupported .npy version %d", major)
	}
	if start+hlen > len(raw) {
		return nil, fmt.Errorf("header of %d bytes runs past the %d-byte file", hlen, len(raw))
	}
	header := string(raw[start : start+hlen])
	descr := descrRe.FindStringSubmatch(header)
	if descr == nil {
		return nil, fmt.Errorf("no descr in header %q", header)
	}
	if f := fortranRe.FindStringSubmatch(header); f == nil || f[1] != "False" {
		return nil, fmt.Errorf("fortran-ordered or unlabelled array in header %q", header)
	}
	shape := shapeRe.FindStringSubmatch(header)
	if shape == nil {
		return nil, fmt.Errorf("no shape in header %q", header)
	}
	a := &Array{Dtype: descr[1], data: raw[start+hlen:]}
	for _, part := range strings.Split(shape[1], ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		d, err := strconv.Atoi(part)
		if err != nil || d < 0 {
			return nil, fmt.Errorf("bad dimension %q in header %q", part, header)
		}
		a.Shape = append(a.Shape, d)
	}
	width, err := itemSize(a.Dtype)
	if err != nil {
		return nil, err
	}
	if want := a.Len() * width; len(a.data) != want {
		return nil, fmt.Errorf("%s array of shape %v needs %d bytes, the file holds %d", a.Dtype, a.Shape, want, len(a.data))
	}
	return a, nil
}

func itemSize(descr string) (int, error) {
	switch descr {
	case "|u1", "|i1", "|b1":
		return 1, nil
	case "<u2", "<i2", "<f2":
		return 2, nil
	case "<u4", "<i4", "<f4":
		return 4, nil
	case "<u8", "<i8", "<f8":
		return 8, nil
	}
	return 0, fmt.Errorf("unsupported dtype %q (little-endian numeric only)", descr)
}

// Float32s returns the elements of an f4 or f8 array as float32.
func (a *Array) Float32s() ([]float32, error) {
	out := make([]float32, a.Len())
	switch a.Dtype {
	case "<f4":
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(a.data[4*i:]))
		}
	case "<f8":
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(a.data[8*i:])))
		}
	default:
		return nil, fmt.Errorf("dtype %s is not a float array", a.Dtype)
	}
	return out, nil
}

// Float64 returns the single element of a scalar (or one-element) float array.
func (a *Array) Float64() (float64, error) {
	if a.Len() != 1 {
		return 0, fmt.Errorf("array of shape %v is not a scalar", a.Shape)
	}
	switch a.Dtype {
	case "<f8":
		return math.Float64frombits(binary.LittleEndian.Uint64(a.data)), nil
	case "<f4":
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(a.data))), nil
	}
	return 0, fmt.Errorf("dtype %s is not a float array", a.Dtype)
}

// Int64s returns the elements of a signed or unsigned integer array.
func (a *Array) Int64s() ([]int64, error) {
	out := make([]int64, a.Len())
	for i := range out {
		switch a.Dtype {
		case "<i8":
			out[i] = int64(binary.LittleEndian.Uint64(a.data[8*i:]))
		case "<u8":
			u := binary.LittleEndian.Uint64(a.data[8*i:])
			if u > math.MaxInt64 {
				return nil, fmt.Errorf("element %d: %d does not fit int64", i, u)
			}
			out[i] = int64(u)
		case "<i4":
			out[i] = int64(int32(binary.LittleEndian.Uint32(a.data[4*i:])))
		case "<u4":
			out[i] = int64(binary.LittleEndian.Uint32(a.data[4*i:]))
		case "|u1":
			out[i] = int64(a.data[i])
		default:
			return nil, fmt.Errorf("dtype %s is not an integer array", a.Dtype)
		}
	}
	return out, nil
}

// Uint8s returns the elements of an integer array that all lie in 0..=255, the
// way the references' pixel arrays are stored (int64 holding byte values; the
// Python tests cast them with astype(uint8)). An element outside the range is
// refused: a silent wrap would hide a reference gone wrong.
func (a *Array) Uint8s() ([]byte, error) {
	v, err := a.Int64s()
	if err != nil {
		return nil, err
	}
	out := make([]byte, len(v))
	for i, x := range v {
		if x < 0 || x > 255 {
			return nil, fmt.Errorf("element %d is %d, outside 0..=255", i, x)
		}
		out[i] = byte(x)
	}
	return out, nil
}
