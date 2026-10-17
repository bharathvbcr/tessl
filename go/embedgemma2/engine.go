//go:build darwin && arm64

package embedgemma2

/*
// The umbrella archive (go/engine, `make -C go test` builds it) is libgusset.a:
// gusset and tessl in one Rust staticlib, so a Go binary links exactly one
// (docs/adoption.md in gusset, R14). A Rust staticlib carries none of the
// system libraries its crates link, so they are named here: what
// `cargo rustc -- --print native-static-libs` lists for the archive (Metal and
// CoreGraphics for the device; AudioToolbox, ImageIO and AVFoundation with
// CoreMedia and CoreVideo for the media decoders; objc for the Objective-C
// runtime objc2 calls into).
//
// Go's build cache keys a cgo package on the text of its flags, not on the
// bytes of the archive they name. go/Makefile puts the archive's hash in
// CGO_CFLAGS for that reason; a bare `go test` after rebuilding the archive
// should pass -count=1 -a.
#cgo LDFLAGS: -L${SRCDIR}/../engine/target/release -lgusset
#cgo darwin LDFLAGS: -framework AudioToolbox -framework AVFoundation -framework CoreFoundation -framework CoreGraphics -framework CoreMedia -framework CoreVideo -framework Foundation -framework ImageIO -framework Metal -lobjc -liconv
#cgo noescape tessl_go_engine_init
#cgo nocallback tessl_go_engine_init
#cgo noescape tessl_go_engine_wire_version
#cgo nocallback tessl_go_engine_wire_version
#include <stdint.h>
int32_t tessl_go_engine_init(void);
uint32_t tessl_go_engine_wire_version(void);
*/
import "C"

import (
	"fmt"
	"sync"

	"github.com/bharathvbcr/tessl/go/embedgemma2/internal/wire"
)

// registered is the outcome of installing the engine on gusset, once.
var registered struct {
	once sync.Once
	err  error
}

// registerEngine installs the engine's operations on gusset's dispatch and
// checks that the archive speaks this package's frames. Registration is a Rust
// API, not a gusset export, so the umbrella exports its own entry point; it
// must run before the first submission, and a submission made without it is
// refused.
func registerEngine() error {
	registered.once.Do(func() {
		if got := uint32(C.tessl_go_engine_wire_version()); got != wire.Version {
			registered.err = fmt.Errorf("embedgemma2: the linked engine archive speaks frame version %d, this package %d: rebuild go/engine (make -C go engine)", got, wire.Version)
			return
		}
		if failed := int(C.tessl_go_engine_init()); failed != 0 {
			registered.err = fmt.Errorf("embedgemma2: %d engine operations did not register on gusset", failed)
		}
	})
	return registered.err
}
