# S12 — Documentation style and information architecture

Sources: Google developer documentation style guide
(`developers.google.com/style/`), AWS style conventions as reflected in AWS
documentation and the AWS blog editorial requirements, and Nielsen's findings on
scannability of web text.

For a teaching artefact, documentation quality *is* product quality. This standard
also carries the discoverability requirement.

---

## Tone, voice, person

- **S12-C01** Conversational and friendly, never frivolous.
- **S12-C02** Active voice; make clear who performs the action.
- **S12-C03** Second person — "you", not "we".
- **S12-C04** Present tense.
- **S12-C05** Write for a global audience: no idiom, no colloquialism, no cultural
  reference that does not translate.
- **S12-C06** Neutral, professional, humble, direct. No hype, no superlatives, no
  marketing language.
- **S12-C07** No pre-announcements of unreleased capability.

## Sentence and structure

- **S12-C08** Conditions before instructions: "To do X, do Y" — not "Do Y to do X".
- **S12-C09** Short sentences. One idea each.
- **S12-C10** Standard American spelling and punctuation, applied consistently.
- **S12-C11** Serial (Oxford) commas.
- **S12-C12** Unambiguous date formats.

## Headings and titles

- **S12-C13** Sentence case for titles and headings.
- **S12-C14** Headings describe content, so they work as a table of contents.
- **S12-C15** Well-formed hierarchy; no skipped levels.
- **S12-C16** Headings are stable and linkable.

## Lists and tables

- **S12-C17** Numbered lists for sequences; bulleted for unordered sets.
- **S12-C18** Parallel grammatical construction within a list.
- **S12-C19** Tables for comparison across a consistent set of attributes; every row
  populated for every column.
- **S12-C20** Table headers describe the column, not the first row's content.

## Links and code

- **S12-C21** Descriptive link text; never "click here" or a bare URL as link text.
- **S12-C22** Every link resolves. A broken relative path in a README is a defect.
- **S12-C23** Code-related strings in code font.
- **S12-C24** Code blocks carry a language tag for syntax highlighting.
- **S12-C25** Commands are copy-pasteable as written.

## Accessibility

- **S12-C26** Write accessibly: meaning does not depend on colour, position, or
  visual formatting alone.
- **S12-C27** Meaningful alt text for any image. *(N/A — no images shipped.)*
- **S12-C28** Tables are simple, without merged cells, so screen readers can parse them.

## Words to avoid

- **S12-C29** No "simply", "just", "easy", "obviously", "of course" — they tell a
  stuck reader the failure is theirs.
- **S12-C30** No "please" in instructions.
- **S12-C31** No non-inclusive terms (see S10-C31).
- **S12-C32** No unexplained jargon or acronym on first use.

## Information architecture — the discoverability requirement

This is the user's explicit requirement: a reader must know where to go by
*eyeballing* the README, not by reading it.

- **S12-C33** The reader learns which directory serves their model family within the
  first screen of the README.
- **S12-C34** A navigation table maps family → directory → what is covered → which
  API. One row per family, scannable in a single pass.
- **S12-C35** The shared-prerequisite pointer (read foundations first) is stated once,
  prominently, not repeated as noise.
- **S12-C36** Each notebook's scope is visible from the README without opening it.
- **S12-C37** Every directory is reachable from the README; no orphan content.
- **S12-C38** Each family notebook links back to the foundations material, so a
  reader who arrives from a search engine can find the shared mechanics.

## Search discoverability

- **S12-C39** The README's first paragraph contains the terms a reader would search
  for: the service name, the endpoint name, the API names, and the model families.
- **S12-C40** Model identifiers appear verbatim, so an exact-match search finds them.
- **S12-C41** Notebook filenames are descriptive and hyphenated, forming readable URLs.
- **S12-C42** Error strings a reader would paste into a search engine appear verbatim
  in the gotcha documentation.
- **S12-C43** The title states the subject plainly, without cleverness.

## Accuracy

- **S12-C44** Every factual claim is either demonstrated by executed output or
  attributed to a named source.
- **S12-C45** Counts, tallies, and inventory figures in prose match reality.
- **S12-C46** No plagiarism: no copied documentation text; behaviour is described
  from observation.
