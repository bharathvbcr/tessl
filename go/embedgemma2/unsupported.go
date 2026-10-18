//go:build !darwin || !arm64

package embedgemma2

// tessl drives Metal on Apple silicon; there is no other target. The package
// fails to build elsewhere with this message instead of failing at the link.
var _ int = "package embedgemma2 builds on darwin/arm64 only: tessl runs on Apple-silicon Metal"
