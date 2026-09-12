## What does this change?

<!-- What can someone do now that they could not before? -->

## Related issue

<!-- Fixes #123 -->

## Checklist

- [ ] `PYTHONPATH=src python3 -m unittest discover -s tests` passes
- [ ] No new dependencies (this package is standard library only, on purpose)
- [ ] No secrets, real hostnames, account IDs, or personal data added
- [ ] Any new example in `data/` is **invented** — see SECURITY.md

## Does this change any measured number?

<!--
Changing data/gold.jsonl, the prompt, the features or the head changes every
figure in docs/JOURNAL.md. If you touched any of those, re-run the affected
measurements and update the journal in the same PR, so the published numbers
always describe the published data. Say "no" if nothing moved.
-->

## Does this change the decision boundary?

<!--
docs/TAXONOMY.md is the specification. If a label in data/ no longer matches
what the taxonomy says, one of the two is wrong — fix that first, in this PR.
-->
