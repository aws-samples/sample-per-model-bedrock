---
name: discover-bedrock-changes
description: How to find Bedrock changes worth a code sample - which sources to read, in what order, and how to tell a real change from an announcement that needs no code. Use in step 2 of the daily cycle.
---

# Finding changes worth a sample

## Start with the service, not the news

Counter-intuitive but it is the cheaper path. Call `bedrock_list_models` for us-east-1,
us-west-2 and eu-central-1, and `bedrock_list_runtime_models` for us-east-1 and us-west-2.
Diff the model IDs against what the repository mentions:

```
Grep for each catalogue ID across *.ipynb, README.md, _shared/, 99-cross-cutting/
```

An ID in the catalogue that appears nowhere in the repository is either a genuine gap or
deliberately out of scope. An ID the repository mentions that has left the catalogue is a
sample that will now fail for a reader — usually more urgent than an addition.

This finds new models in one step, with no reading, and it cannot be fooled by a
mis-dated announcement.

## Then read the feeds, filtered by date

Use `sources_feed` on the RSS feeds in your system prompt, with `since` set to the
previous checkpoint's date — you do not need to re-read what you read yesterday. Use
`sources_feed` rather than `WebFetch` here: `WebFetch` summarises through a model and can
omit an entry, and an omission at this step is indistinguishable from a quiet day.
`WebFetch` is the right tool once you are reading a specific post you have decided
matters.

**Read `window_covered` in each result.** A feed holds a fixed number of entries, so if
it does not reach back to your `since` date, the days before its oldest entry were not
checked and an empty result proves nothing. Close the gap with `sources_fetch` on the
dated What's New index page, or record in your checkpoint that the window was not fully
covered so tomorrow's run knows to look further back.

For each candidate, answer one question before spending anything else on it: **would a
developer have to write different code because of this?**

| Announcement | Needs a sample? |
|---|---|
| New model on Bedrock | Yes, if it is a text or multimodal chat model |
| Model reaches a new Region | Only if the repository asserts a Region list that is now wrong |
| New API parameter or request shape | Yes |
| Price change | No |
| Console feature | No |
| New quota defaults | No, unless a sample would now throttle |
| Model deprecation | Yes, and urgently, if the repository still uses it |

## Then look for behaviour nobody wrote up

This is where the highest-value findings come from, and the reason re:Post is on the list.
A customer reporting "this used to work" is often the first sign that an ID moved or an
API narrowed. Search re:Post's `amazon-bedrock` tag for the last week or two, and treat a
plausible report as a hypothesis to **test against the live service**, never as a fact to
publish.

## The discrepancy is itself the finding

When an announcement and the catalogue disagree, that is worth recording, not smoothing
over. Real examples from this repository's history:

- Model cards listed an API as supported that both endpoints refused.
- A documentation page's prose said `endpoint` must be `v1/chat/completions` while its own
  code examples used `/v1/chat/completions`.
- A model card's In-Region URL column said `N/A` for a model the docs elsewhere implied
  worked there.

A sample that shows the reader what the service actually does, and notes where the
documentation differs, is more useful than one that repeats the documentation.

## What to hand to the next step

A short list. For each item: what changed, the source URL, what you verified against the
live service, and which files in the repository are affected. If that list is empty, say
so and go to step 9 — that is a complete and successful cycle.
