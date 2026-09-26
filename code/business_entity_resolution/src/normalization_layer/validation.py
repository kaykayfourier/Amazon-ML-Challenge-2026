"""
Validation harness for the normalization layer.

Every normalization section ends with a checkpoint. A checkpoint collects
issues rather than raising on the first one, so a single run tells you
everything that is wrong instead of the first thing.

    chk = Check("missingness/S2")
    chk.equal(len(out), len(df), "row count preserved")
    chk.absent(out["name_norm"], {"nan", "none"}, "no coerced-null strings")
    chk.report()          # prints; raises if strict and anything failed
"""


class Check:
    def __init__(self, label, strict=True):
        self.label = label
        self.strict = strict
        self.issues = []
        self.passed = 0

    # -- primitives --------------------------------------------------------

    def ok(self, condition, msg, detail=""):
        if condition:
            self.passed += 1
        else:
            self.issues.append(f"{msg}{f' -- {detail}' if detail else ''}")
        return self

    def equal(self, got, want, msg):
        return self.ok(got == want, msg, f"got {got!r}, want {want!r}")

    def between(self, value, lo, hi, msg):
        return self.ok(lo <= value <= hi, msg, f"got {value!r}, expected [{lo}, {hi}]")

    # -- series-level ------------------------------------------------------

    def no_nulls(self, series, msg=None):
        n = series.isna().sum()
        return self.ok(n == 0, msg or f"{series.name}: no nulls", f"{n} nulls")

    def absent(self, series, forbidden, msg=None):
        """No value in `forbidden` appears in the series (case-insensitive)."""
        hits = series.str.lower().isin({f.lower() for f in forbidden})
        n = int(hits.sum())
        return self.ok(n == 0, msg or f"{series.name}: no forbidden literals",
                       f"{n} rows, e.g. {series[hits].head(3).tolist()}")

    def is_bool(self, series, msg=None):
        return self.ok(series.dtype == bool, msg or f"{series.name}: boolean dtype",
                       f"dtype is {series.dtype}")

    def rate(self, series, lo, hi, msg=None):
        """Fraction of True in a boolean series falls within [lo, hi]."""
        r = float(series.mean())
        return self.ok(lo <= r <= hi, msg or f"{series.name}: rate in range",
                       f"got {r:.4f}, expected [{lo}, {hi}]")

    # -- output ------------------------------------------------------------

    def report(self):
        if self.issues:
            body = "\n".join(f"  - {i}" for i in self.issues)
            msg = f"[{self.label}] {len(self.issues)} issue(s):\n{body}"
            if self.strict:
                raise AssertionError(msg)
            print(msg)
        else:
            print(f"[{self.label}] ok ({self.passed} checks)")
        return not self.issues