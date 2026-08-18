# LTX-2.5's licence, read properly

Written on 2026-08-18 after the audio path came up, because the remaining work
is integration and integration is the expensive part to do twice. The full
Agreement ships **inside the checkpoint** — `metadata()["license"]`, 34,562
characters, dated 11 August 2026 — so this is read off the weights we actually
have rather than a web page.

None of what follows is legal advice. It is the set of questions a lawyer would
need to answer, with the text that raises them.

## The clause that decides whether this ships at all

Attachment A, Prohibited Uses, **clause 20**:

> To use LTX-2.x or Derivatives of LTX-2.x in any product, service, or
> application that directly competes with Licensor's commercial products or
> services, or is designed to replace or substitute Licensor's offerings in the
> market, without obtaining a separate commercial license from Licensor.

Lightricks sells **LTX Studio**, an AI video generation product. H3ddle is an
AI video generation app. Whether that is "directly competes" is a judgement
call, but it is not a stretch, and the restriction is **not** gated on revenue:
§2.1 grants the licence "subject to the restrictions set forth in Attachment
A", so clause 20 binds every licensee including a hobbyist.

**This is the gate.** Everything below matters only if it clears.

## The revenue threshold, which is a separate question

§2.1 grants a royalty-free licence for any purpose, except that **Entities with
annual revenues of at least $10,000,000** ("Commercial Entities") must obtain a
paid Commercial Use Agreement (`ltxv-licensing@lightricks.com`).

§2.2 lets a Commercial Entity skip that *only* for a Non-Commercial Purpose,
defined narrowly: personal hobby/research use, or "testing, evaluation, or
non-commercial research and development in a non-production or development
environment". It then closes the obvious doors — use "(b) in direct
interactions with or that has impact on end users" is explicitly **not** a
Non-Commercial Purpose.

So a shipped app is production use by definition. Below $10M this costs
nothing; at or above it, a paid agreement is required regardless of clause 20.

## Watermarking: nothing to preserve, but an obligation to meet

The transparency clause forbids removing, disabling, altering or circumventing
"any safety or security measures, disclosures, metadata, watermarking, content
provenance, latent disclosure, or other transparency features … included or
embedded within LTX-2.x", and lets Licensor **revoke the licence at its sole
discretion** if it believes you have.

Audited: `ltx_core` contains **no** watermarking, C2PA, provenance, signing or
fingerprinting code — zero matches across the package. So there is nothing
embedded to preserve and nothing being circumvented by this port. That is the
good news and it is worth having checked rather than assumed, because the
penalty clause is unusually sharp.

The obligation that *does* land on us is the next sentence: the licensee is
"solely responsible for any transparency, disclosure, marking, or labeling
obligations applicable to you under AI Regulations … including any obligation
to disclose that content is artificially generated". That is a product feature
— label LTX outputs as AI-generated — and it is cheap. It should be built
whether or not it is strictly required, and it is not a porting blocker.

## Redistribution

§3 permits redistribution and requires shipping a complete copy of the
Agreement (§3.2). Since the Agreement is already in the checkpoint metadata,
mirroring the weights carries its own licence — but the app should surface it
too, as it already does for Stability's terms on the SFX package.

## What to do with this

1. **Answer clause 20 before writing integration code.** Either a reading that
   H3ddle does not directly compete, or an email to
   `ltxv-licensing@lightricks.com`. It is one message and it de-risks days of
   work; the alternative is building the vertical slice and discovering the
   answer afterwards.
2. **Check the revenue threshold** — likely moot, but it is a yes/no.
3. **Build the AI-generated label** regardless. It is owed under AI Regulations
   independent of this licence, and it is small.

## How this compares to what already ships

Worth stating plainly, because it changes the LTX-vs-H3 calculus beyond
quality: the SFX and music packages carry Stability's terms and the speech
package Qwen's, and neither carries a competing-product prohibition. LTX is the
only engine considered so far whose licence contains a clause that could
prohibit the app's core purpose outright.
