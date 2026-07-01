---
name: meeting-narrative
description: >
  Turn an earshot meeting (diarized transcript.json/transcript.md) plus any
  take-notes file into a single narrative write-up of the meeting: who said
  what, decisions, action items, and significant realizations. Use when the
  user asks to "write up the meeting", "create a meeting narrative",
  "narrate this meeting", or points at an earshot meeting folder and wants a
  readable account rather than a raw transcript.
user-invocable: true
argument-hint: "[path to earshot meeting directory]"
allowed-tools: Read, Write, Glob, Grep
---

# Meeting Narrative

Produce one readable document that tells the story of a meeting, synthesized
from an earshot machine transcript and, when available, the notes the user
took during the meeting with `/take-notes`. The transcript supplies coverage
(everything said, with speakers and timestamps); the user's notes supply
judgment (what actually mattered). The narrative merges both.

## Inputs

1. **The meeting directory** (given as the argument, or ask for it). It is an
   earshot output folder like `.../YYYY-MM-DD_HHMMSS_<title-slug>/` containing:
   - `transcript.json` — the ground truth. Fields that matter:
     - `title`, `generated_at`, `diarized_at`
     - `segments[]`: each has `speaker`, `channel` (`near` = the user,
       `far` = everyone else), `start`/`end` (seconds), `text`
     - `diarization.labels` and `diarization.identification` — which speakers
       are real names vs still-anonymous `REMOTE-N`
   - `transcript.md` — the same content, human-readable. Use it for quick
     orientation; prefer `transcript.json` when they disagree.
   - `meeting.json` — capture metadata (`title`, `started_at`, `ended_at`).
   - `notes.md` — machine-generated summary, may exist. Treat it as a weak
     prior only; your narrative supersedes it. Never copy from it.

2. **The user's own notes**, taken live with `/take-notes`. Find them: glob
   `meeting-notes/*.md` in the workspace and match on the meeting date (from
   `meeting.json.started_at` or the folder name) and title similarity. If one
   plausible file matches, use it and say so; if several could match or none
   do, ask the user rather than guessing. These notes follow a known shape:
   a header (Date/Attendees/Purpose), then `## Notes` (chronological,
   speaker-attributed entries), `## Action Items` (owner-tagged checkboxes),
   and `## Open Questions`.

## Before writing: reconcile speakers

- `YOU` in the transcript is the user. Use their real name if the notes or
  prior narratives reveal it; otherwise keep "You" consistently.
- If `REMOTE-N` labels remain, try to resolve them from evidence before
  writing: the attendee list in the user's notes, self-introductions in the
  transcript text ("this is Sara"), or vocative addressing ("thanks, Sara").
  State each inference and its evidence to the user and get a confirmation
  for any speaker you plan to name this way. If the user can't confirm,
  keep the neutral label and suggest running `earshot attribute <dir>`
  first, which names speakers by ear and teaches the voice library.
- Never silently guess a name. A wrong attribution is worse than REMOTE-2.

## Writing the narrative

This is a re-telling, not a transcript. Write it the way a sharp chief of
staff would brief someone who missed the meeting.

- **Structure by topic, not by time.** Group the discussion into its natural
  threads (usually 2-5), even if the conversation bounced between them.
  Within a thread, keep chronological order.
- **Attribute positions, not sentences.** "Dan pushed for shipping the tag
  in this release; Sara wanted it behind a flag until the audit lands" beats
  quoting either of them. Quote verbatim (short, in quotes) only when the
  exact wording carries weight: a commitment, a number, a memorable framing.
- **The transcript is ASR output.** Expect mis-transcriptions, especially of
  names, acronyms, and product terms. Repair them when the correct term is
  clear from the user's notes, the meeting title, or surrounding context.
  Do not repair into a guess; if a garbled passage matters and can't be
  confidently reconstructed, paraphrase what's clear and flag the gap with
  the timestamp so the user can listen to that moment.
- **The user's notes carry emphasis and corrections.** When the notes and
  the transcript conflict on substance, prefer the notes (the user was in
  the room) and note the discrepancy in one clause. When the notes flag
  something the transcript barely touches, that's a signal it mattered:
  give it weight in the narrative.
- **Don't invent.** Every claim in the narrative must trace to the
  transcript or the notes. No fabricated agreements, owners, or dates.
- Timestamps: use them sparingly, as `[MM:SS]` anchors on decisions and on
  anything flagged for follow-up listening. Whisper timestamps can drift
  slightly; treat them as approximate.

## Output document

Write `narrative.md` into the meeting directory:

```markdown
# <Meeting title, humanized>

**Date**: YYYY-MM-DD  |  **Participants**: Name, Name, ... (note any
still-unidentified speakers)  |  **Sources**: transcript + your notes /
transcript only

One-paragraph overview: why the meeting happened and the single most
important outcome.

## The discussion

Topic-organized narrative prose. Who argued what, how positions moved,
where the group landed. A few short verbatim quotes where wording matters.

## Decisions

- The decision, who drove it, and any conditions attached.

## Action items

- [ ] **Owner** — the commitment, with due date if one was stated. Merge
  duplicates between notes and transcript; the notes' owner wins on conflict.

## Significant realizations

- Insights, reframings, or surprises that changed the group's thinking.
  This is for the "we realized X" moments, not routine information exchange.

## Open questions

- Unresolved items, each with one line of context on why it matters.
```

Omit a section (except The discussion) rather than padding it. If the user's
notes contributed, say which file was used in **Sources**.

After writing, tell the user the path and give a two-sentence version of the
meeting, then mention anything you flagged (unresolved speakers, garbled
passages worth a listen).

## Writing style

- Direct, natural prose. No corporate filler, no "the team then discussed".
- No em-dashes. Use commas, periods, or semicolons.
- Complete sentences in the narrative; terse fragments are fine in the
  Decisions/Action items/Open questions lists.
- Write like a colleague who was there, not like a summarizer. Specifics
  over abstractions: names, numbers, dates, system names.
