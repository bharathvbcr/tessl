package embedgemma2

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestUnitErrorMessageCarriesTesslsTextVerbatim(t *testing.T) {
	e := &Error{Op: "EmbedTexts", Status: StatusError, Message: "no prompt named \"nope\" (the snapshot has: query, document)"}
	if got := e.Error(); got != "embedgemma2: EmbedTexts: no prompt named \"nope\" (the snapshot has: query, document)" {
		t.Fatalf("got %q", got)
	}
}

func TestUnitErrorMatchesSentinelsByStatus(t *testing.T) {
	for name, tc := range map[string]struct {
		err            *Error
		panic, poison  bool
		noModel, other bool
	}{
		"plain refusal":      {err: &Error{Status: StatusError}},
		"panic":              {err: &Error{Status: StatusPanic}, panic: true},
		"poisoned":           {err: &Error{Status: StatusError, Poisoned: true}, poison: true},
		"no model":           {err: &Error{Status: StatusNoModel}, noModel: true},
		"panic and poisoned": {err: &Error{Status: StatusPanic, Poisoned: true}, panic: true, poison: true},
	} {
		wrapped := fmt.Errorf("outer: %w", tc.err)
		for sentinel, want := range map[error]bool{ErrPanic: tc.panic, ErrPoisoned: tc.poison, ErrNoModel: tc.noModel, ErrInvalidArgument: false, ErrClosed: false, ErrProtocol: false} {
			if got := errors.Is(wrapped, sentinel); got != want {
				t.Errorf("%s: errors.Is(%v) = %v, want %v", name, sentinel, got, want)
			}
		}
		var back *Error
		if !errors.As(wrapped, &back) || back != tc.err {
			t.Errorf("%s: errors.As lost the *Error", name)
		}
	}
}

func TestUnitArgErrorNamesOperationFieldAndReason(t *testing.T) {
	err := argErr("EmbedImages", "images[2]", "holds %d bytes, but %dx%dx%d needs %d", 5, 2, 2, 3, 12)
	var ae *ArgError
	if !errors.As(err, &ae) || ae.Op != "EmbedImages" || ae.Field != "images[2]" || !strings.HasPrefix(ae.Reason, "holds 5 bytes") {
		t.Fatalf("got %#v", err)
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Error("an ArgError does not match ErrInvalidArgument")
	}
	if errors.Is(err, ErrPanic) || errors.Is(err, ErrClosed) {
		t.Error("an ArgError matches a sentinel it should not")
	}
	if want := "embedgemma2: EmbedImages: images[2]: holds 5 bytes, but 2x2x3 needs 12"; err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestUnitProtocolErrorMatchesErrProtocolOnly(t *testing.T) {
	err := protocolErr("Tokenize", "%d offsets", 3)
	if !errors.Is(err, ErrProtocol) || errors.Is(err, ErrInvalidArgument) || errors.Is(err, ErrPanic) {
		t.Fatalf("got %v", err)
	}
	if !strings.Contains(err.Error(), "3 offsets") {
		t.Errorf("got %q", err.Error())
	}
}
