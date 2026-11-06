package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"
)

// scriptModel wraps the fake and rewrites what Tokenize answers.
type scriptModel struct {
	*fakeModel
	tok func(texts []string, prompt string, real [][]uint32) [][]uint32
}

func (s *scriptModel) Tokenize(ctx context.Context, texts []string, prompt string) ([][]uint32, error) {
	ids, err := s.fakeModel.Tokenize(ctx, texts, prompt)
	if err != nil {
		return nil, err
	}
	return s.tok(texts, prompt, ids), nil
}

func newScripted(t *testing.T, m *scriptModel) (*server, string) {
	t.Helper()
	var buf bytes.Buffer
	orig := log.Writer()
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(orig) })
	s, err := newServer(context.Background(), m, m.promptSet(), testLimits(), modelInfo{Size: 1, Digest: "r", ModifiedAt: time.Unix(0, 0).UTC()}, func(error) {})
	if err != nil {
		t.Fatal(err)
	}
	return s, buf.String()
}

// A reply with the wrong number of sequences means no EOS (fail closed to the
// plain cut); characterized here, with the log line asserted.
func TestUnitEOSProbeMalformedReplyIsNoEOS(t *testing.T) {
	for _, n := range []int{0, 1, 3, 5} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			m := &scriptModel{fakeModel: newFake(), tok: func(texts []string, _ string, real [][]uint32) [][]uint32 {
				out := make([][]uint32, n)
				for i := range out {
					out[i] = []uint32{2, 7, 1}
				}
				return out
			}}
			s, logged := newScripted(t, m)
			if s.hasEOS {
				t.Fatalf("hasEOS with %d sequences for %d probes", n, len(eosProbes))
			}
			if !strings.Contains(logged, "no end-of-sequence id") {
				t.Fatalf("no log line: %q", logged)
			}
		})
	}
}

// A single short sequence among otherwise agreeing ones disqualifies the probe,
// as does an empty one; a tokenizer error is an error, not "no EOS".
func TestUnitEOSProbeShortSequenceAndErrors(t *testing.T) {
	for i := range eosProbes {
		m := &scriptModel{fakeModel: newFake(), tok: func(texts []string, _ string, real [][]uint32) [][]uint32 {
			out := make([][]uint32, len(real))
			for j := range out {
				out[j] = []uint32{2, 7, 1}
			}
			out[i] = []uint32{1}
			return out
		}}
		if s, _ := newScripted(t, m); s.hasEOS {
			t.Fatalf("a one-id sequence at probe %d still gave an EOS", i)
		}
	}
	f := newFake()
	f.tokenizeErr = errors.New("engine down")
	if _, err := newServer(context.Background(), f, f.promptSet(), testLimits(), modelInfo{}, nil); err == nil || !strings.Contains(err.Error(), "engine down") {
		t.Fatalf("tokenizer failure: %v", err)
	}
}

// Success is silent: no log line in the normal case.
func TestUnitEOSProbeLogsOnlyWhenThereIsNoEOS(t *testing.T) {
	var buf bytes.Buffer
	orig := log.Writer()
	log.SetOutput(&buf)
	defer log.SetOutput(orig)
	newHarness(t, newFake(), testLimits())
	if buf.Len() != 0 {
		t.Fatalf("logged on success: %q", buf.String())
	}
	f := newFake()
	f.noEOS = true
	newHarness(t, f, testLimits())
	if got := buf.String(); strings.Count(got, "\n") != 1 || !strings.Contains(got, "plain first 64 ids") {
		t.Fatalf("log: %q", got)
	}
}

// Truncating to one id with an EOS yields exactly [eos], the cut the device
// test's reference (ids[:limit-1] + eos) gives at limit 1.
func TestUnitTruncateToOneIDKeepsOnlyEOS(t *testing.T) {
	for _, limit := range []int{1, 2} {
		fake := newFake()
		fake.maxTokens = limit
		h := newHarness(t, fake, testLimits())
		status, v := h.post("/api/embed", fmt.Sprintf(`{"model":%q,"input":"hello world"}`, baseName+":query"))
		if status != http.StatusOK {
			t.Fatalf("%d %v", status, v)
		}
		cut := fake.lastIDs[0]
		if len(cut) != limit || cut[limit-1] != 1 {
			t.Fatalf("limit %d: cut %v", limit, cut)
		}
	}
}

// A sequence that does not end in the EOS (a text that carries its own <bos>
// gets no <eos>) is cut to the plain first ids, not given an invented EOS.
func TestUnitTruncateSequenceWithoutEOSIsPlainCut(t *testing.T) {
	m := &scriptModel{fakeModel: newFake()}
	m.tok = func(texts []string, prompt string, real [][]uint32) [][]uint32 {
		out := make([][]uint32, len(real))
		for i, r := range real {
			if strings.HasPrefix(texts[i], "<bos>") {
				out[i] = r[:len(r)-1]
			} else {
				out[i] = r
			}
		}
		return out
	}
	m.maxTokens = 6
	s, _ := newScripted(t, m)
	if !s.hasEOS || s.eosID != 1 {
		t.Fatalf("hasEOS %v eos %d", s.hasEOS, s.eosID)
	}
	ids := [][]uint32{{2, 10, 11, 12, 13, 14, 15, 16, 17}, {2, 10, 11, 12, 13, 14, 15, 16, 1}}
	got, _, herr := s.fit(ids, true)
	if herr != nil {
		t.Fatal(herr)
	}
	if !slices.Equal(got[0], []uint32{2, 10, 11, 12, 13, 14}) {
		t.Fatalf("no-eos sequence cut to %v", got[0])
	}
	if !slices.Equal(got[1], []uint32{2, 10, 11, 12, 13, 1}) {
		t.Fatalf("eos sequence cut to %v", got[1])
	}
}

// The probe must not be fooled by a default prompt that is longer than the
// context limit, nor by MaxTokens below the probe length: Tokenize applies no
// limit (capi: tokenize_texts(.., None)), and the fake models that.
func TestUnitEOSProbeIgnoresMaxTokensAndLongPrompt(t *testing.T) {
	for _, mt := range []int{1, 2, 3} {
		fake := newFake()
		fake.maxTokens = mt
		fake.defaultPrompt = strings.Repeat("p", 500)
		h := newHarness(t, fake, testLimits())
		if !h.srv.hasEOS || h.srv.eosID != 1 {
			t.Fatalf("maxTokens %d: hasEOS %v eos %d", mt, h.srv.hasEOS, h.srv.eosID)
		}
	}
}
