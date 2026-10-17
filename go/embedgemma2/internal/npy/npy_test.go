package npy

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

// file builds a version-1 .npy file around data.
func file(descr, shape string, data []byte) []byte {
	header := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shape)
	for (10+len(header)+1)%64 != 0 {
		header += " "
	}
	header += "\n"
	out := []byte("\x93NUMPY\x01\x00")
	out = binary.LittleEndian.AppendUint16(out, uint16(len(header)))
	out = append(out, header...)
	return append(out, data...)
}

func f32le(v ...float32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

func i64le(v ...int64) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint64(b, uint64(x))
	}
	return b
}

func TestUnitParseFloatMatrix(t *testing.T) {
	a, err := Parse(file("<f4", "(2, 3)", f32le(1, 2, 3, 4, 5, 6)))
	if err != nil {
		t.Fatal(err)
	}
	if a.Dtype != "<f4" || len(a.Shape) != 2 || a.Shape[0] != 2 || a.Shape[1] != 3 || a.Len() != 6 {
		t.Fatalf("got %+v", a)
	}
	v, err := a.Float32s()
	if err != nil || v[5] != 6 {
		t.Fatalf("%v %v", v, err)
	}
}

func TestUnitParseOneDimensionalAndScalarShapes(t *testing.T) {
	a, err := Parse(file("<f4", "(3,)", f32le(7, 8, 9)))
	if err != nil || a.Len() != 3 {
		t.Fatalf("%+v %v", a, err)
	}
	s, err := Parse(file("<f8", "()", binary.LittleEndian.AppendUint64(nil, math.Float64bits(4.5))))
	if err != nil {
		t.Fatal(err)
	}
	if v, err := s.Float64(); err != nil || v != 4.5 {
		t.Fatalf("scalar %v %v", v, err)
	}
	if _, err := a.Float64(); err == nil {
		t.Fatal("a 3-element array read as a scalar")
	}
}

func TestUnitIntegerPixelsConvertToBytesOnlyInRange(t *testing.T) {
	a, err := Parse(file("<i8", "(4,)", i64le(0, 1, 254, 255)))
	if err != nil {
		t.Fatal(err)
	}
	b, err := a.Uint8s()
	if err != nil || len(b) != 4 || b[3] != 255 {
		t.Fatalf("%v %v", b, err)
	}
	bad, _ := Parse(file("<i8", "(2,)", i64le(3, 256)))
	if _, err := bad.Uint8s(); err == nil || !strings.Contains(err.Error(), "outside 0..=255") {
		t.Fatalf("got %v", err)
	}
	neg, _ := Parse(file("<i8", "(1,)", i64le(-1)))
	if _, err := neg.Uint8s(); err == nil {
		t.Fatal("-1 wrapped to a byte")
	}
}

func TestUnitMalformedFilesAreRefusedByName(t *testing.T) {
	for name, tc := range map[string]struct {
		raw  []byte
		want string
	}{
		"no magic":      {[]byte("not numpy at all"), "no magic"},
		"short data":    {file("<f4", "(4,)", f32le(1, 2)), "needs 16 bytes"},
		"long data":     {file("<f4", "(1,)", f32le(1, 2)), "needs 4 bytes"},
		"big endian":    {file(">f4", "(1,)", f32le(1)), "unsupported dtype"},
		"object":        {file("|O", "(1,)", f32le(1)), "unsupported dtype"},
		"bad dimension": {file("<f4", "(x,)", nil), "bad dimension"},
	} {
		if _, err := Parse(tc.raw); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want %q", name, err, tc.want)
		}
	}
	fortran := []byte(strings.Replace(string(file("<f4", "(1,)", f32le(1))), "False", "True ", 1))
	if _, err := Parse(fortran); err == nil || !strings.Contains(err.Error(), "fortran") {
		t.Errorf("fortran order: got %v", err)
	}
	truncated := file("<f4", "(1,)", f32le(1))[:12]
	if _, err := Parse(truncated); err == nil {
		t.Error("a file cut inside its header parsed")
	}
}

func TestUnitFloat8ReadsAsFloat32AndWrongKindsAreRefused(t *testing.T) {
	a, _ := Parse(file("<f8", "(2,)", binary.LittleEndian.AppendUint64(binary.LittleEndian.AppendUint64(nil, math.Float64bits(0.25)), math.Float64bits(-2))))
	v, err := a.Float32s()
	if err != nil || v[0] != 0.25 || v[1] != -2 {
		t.Fatalf("%v %v", v, err)
	}
	if _, err := a.Int64s(); err == nil {
		t.Fatal("floats read as integers")
	}
	i, _ := Parse(file("<i8", "(1,)", i64le(3)))
	if _, err := i.Float32s(); err == nil {
		t.Fatal("integers read as floats")
	}
}
