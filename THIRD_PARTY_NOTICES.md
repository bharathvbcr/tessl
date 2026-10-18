# Third-party notices

## libjpeg / libjpeg-turbo (`src/media/jpeg/`)

**This software is based in part on the work of the Independent JPEG Group.**

`tessl::media::jpeg` is a Rust decoder written to give the same bytes as
libjpeg-turbo 3.1.3 (the library `torchvision.io.decode_image` links). It contains no
libjpeg C source. It follows the library's control flow and reproduces its integer arithmetic:
the marker reader, the Huffman, progressive and arithmetic entropy decoders, the coefficient
controller, the accurate integer inverse DCT (`jidctint.c`, and the 16-bit-lane behaviour of
libjpeg-turbo's 64-bit ARM NEON version on corrupt blocks), the fancy chroma upsampler and the
YCbCr colour converter, the block smoothing of `jdcoefct.c` (`smooth.rs`: libjpeg-turbo's
5x5 estimator, whose integer weight tables were measured from the reference's output on
synthetic progressive files, not copied) and the lossless decoder of libjpeg-turbo 3.x
(`jdlossls.c`, `jdlhuff.c`, `jddiffct.c`, `jdpred.c`, `jdscale.c`; `lossless.rs`). The arithmetic decoder's probability state table (`ARITAB` in
`src/media/jpeg/arith.rs`) holds the same values as the library's `jpeg_aritab`, which is
Table D.3 of ITU-T T.81. These are works derived from or based on the IJG code, so the IJG
License's conditions apply to them. Its text, from libjpeg-turbo's `README.ijg`, is reproduced
unaltered below. libjpeg-turbo's own `LICENSE.md` says its SIMD code is under the zlib License,
and that the IJG License subsumes it in the context of the libjpeg API library.

The reference outputs in `tests/fixtures/jpeg/` were produced by running libjpeg-turbo through
torchvision (`scripts/gen_jpeg_fixtures.py`).

### The IJG License (from `README.ijg`, LEGAL ISSUES)

```text
In plain English:

1. We don't promise that this software works.  (But if you find any bugs,
   please let us know!)
2. You can use this software for whatever you want.  You don't have to pay us.
3. You may not pretend that you wrote this software.  If you use it in a
   program, you must acknowledge somewhere in your documentation that
   you've used the IJG code.

In legalese:

The authors make NO WARRANTY or representation, either express or implied,
with respect to this software, its quality, accuracy, merchantability, or
fitness for a particular purpose.  This software is provided "AS IS", and you,
its user, assume the entire risk as to its quality and accuracy.

This software is copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
All Rights Reserved except as specified below.

Permission is hereby granted to use, copy, modify, and distribute this
software (or portions thereof) for any purpose, without fee, subject to these
conditions:
(1) If any part of the source code for this software is distributed, then this
README file must be included, with this copyright and no-warranty notice
unaltered; and any additions, deletions, or changes to the original files
must be clearly indicated in accompanying documentation.
(2) If only executable code is distributed, then the accompanying
documentation must state that "this software is based in part on the work of
the Independent JPEG Group".
(3) Permission for use of this software is granted only if the user accepts
full responsibility for any undesirable consequences; the authors accept
NO LIABILITY for damages of any kind.

These conditions apply to any software derived from or based on the IJG code,
not just to the unmodified library.  If you use our work, you ought to
acknowledge us.

Permission is NOT granted for the use of any IJG author's name or company name
in advertising or publicity relating to this software or products derived from
it.  This software may be referred to only as "the Independent JPEG Group's
software".

We specifically permit and encourage the use of this software as the basis of
commercial products, provided that all warranty or liability claims are
assumed by the product vendor.
```
