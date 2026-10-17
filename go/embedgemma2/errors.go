package embedgemma2

import (
	"errors"
	"fmt"
)

// Status is the status code tessl's C ABI returned for a failed call.
type Status int

const (
	// StatusError is TESSL_ERR: tessl refused or failed the call, and its
	// message says why. The model stays usable unless Error.Poisoned is set.
	StatusError Status = 1
	// StatusPanic is TESSL_PANIC: a bug in tessl, caught at its boundary. The
	// model is closed to further calls; Close it and Load again.
	StatusPanic Status = 2
	// StatusNoModel is not a tessl status: the engine worker held no usable
	// model. It follows a StatusPanic or a poisoned runtime on the same model,
	// or a worker replaced after a panic.
	StatusNoModel Status = 3
)

var (
	// ErrInvalidArgument matches every refusal made in Go before a call
	// crosses into Rust: a count, length, range or encoding that cannot be
	// right. The *ArgError carries which.
	ErrInvalidArgument = errors.New("embedgemma2: invalid argument")
	// ErrClosed matches a call on a Model after Close.
	ErrClosed = errors.New("embedgemma2: model is closed")
	// ErrPanic matches an *Error whose status is StatusPanic, and a panic the
	// engine's gusset firewall caught: either way the model is unusable.
	ErrPanic = errors.New("embedgemma2: tessl panicked")
	// ErrPoisoned matches an *Error that reports the model's runtime poisoned
	// (an encode, submit or GPU fault latched it; every later call is
	// refused). Close the model and Load it again.
	ErrPoisoned = errors.New("embedgemma2: the model's runtime is poisoned")
	// ErrNoModel matches an *Error with StatusNoModel.
	ErrNoModel = errors.New("embedgemma2: the engine holds no usable model")
	// ErrProtocol matches a failure of the Go/Rust frame itself: a reply this
	// package cannot read, or a request the engine could not decode. It is a
	// bug in this package or a stale engine archive, never a caller's input.
	ErrProtocol = errors.New("embedgemma2: engine protocol error")
)

// Error is a failure tessl reported. Message is tessl's text, verbatim.
type Error struct {
	// Op is the Go method that made the call.
	Op string
	// Status is tessl's status code.
	Status Status
	// Message is the message tessl wrote for the call.
	Message string
	// Poisoned reports that the model's runtime latched a fault and refuses
	// all further work.
	Poisoned bool
}

func (e *Error) Error() string {
	return fmt.Sprintf("embedgemma2: %s: %s", e.Op, e.Message)
}

// Is makes errors.Is(err, ErrPanic), ErrPoisoned and ErrNoModel match by
// status.
func (e *Error) Is(target error) bool {
	switch target {
	case ErrPanic:
		return e.Status == StatusPanic
	case ErrPoisoned:
		return e.Poisoned
	case ErrNoModel:
		return e.Status == StatusNoModel
	}
	return false
}

// ArgError is a refusal made in Go, before anything crossed into Rust.
type ArgError struct {
	// Op is the Go method.
	Op string
	// Field names the argument, with an index when one item of a list is at
	// fault ("images[2]").
	Field string
	// Reason says what is wrong with it.
	Reason string
}

func (e *ArgError) Error() string {
	return fmt.Sprintf("embedgemma2: %s: %s: %s", e.Op, e.Field, e.Reason)
}

// Is makes errors.Is(err, ErrInvalidArgument) match.
func (e *ArgError) Is(target error) bool { return target == ErrInvalidArgument }

func argErr(op, field, format string, args ...any) error {
	return &ArgError{Op: op, Field: field, Reason: fmt.Sprintf(format, args...)}
}

func protocolErr(op, format string, args ...any) error {
	return fmt.Errorf("embedgemma2: %s: %w: %s", op, ErrProtocol, fmt.Sprintf(format, args...))
}
