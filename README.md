# Meltdown Pattern Predictor

**Live:** https://meltdown-pattern-predictor-adahacks.vercel.app/

> A smart tracking tool that helps caregivers of autistic children identify what triggers meltdowns—so families can plan better, stress less, and support their kids with confidence.

Built at **Ada Hacks**.

---

## The Problem

Caregivers of autistic children often feel like they're navigating meltdowns in the dark. A difficult day happens, and in the moment it's hard to pinpoint *why*—Was it the lack of sleep? The loud environment? The change in routine? The late-night screen time?

Without data, it's guesswork. Families replay events in their heads, wonder what they could have done differently, and hope to spot patterns before the next meltdown. But human memory is unreliable, and subtle triggers add up in ways that are easy to miss.

**The result:** Exhausted caregivers, repeated meltdowns, and missed opportunities to create calmer, more predictable environments that support autistic children's well-being.

---

## Our Solution

**Meltdown Pattern Predictor** turns guesswork into insight. Caregivers log simple daily factors—sleep, sensory environment, routine changes, late meals, screen time—and whether a meltdown occurred. Our tool analyzes the data and surfaces which conditions correlate most strongly with meltdowns.

### What We Built

- **Daily Logging** – Quick, structured entries for sleep hours, sensory environment (noise level), late sugar, late screens, routine changes, late meals, and meltdown occurrence
- **Pattern Analysis** – After a few logs, the app computes meltdown rates across each factor and ranks the most likely triggers
- **Data-Driven Insights** – No more relying on memory. See exactly which conditions preceded meltdowns
- **Privacy-First** – Everything stays in your browser. No accounts, no servers, no data leaving your device
- **Actionable** – Use the insights to plan routines, avoid known triggers, and create more supportive environments

### Who It's For

Parents, guardians, and caregivers of children on the autism spectrum who want to understand what contributes to meltdowns—not to blame, but to build better days.

---

## Getting Started

**Local:**
1. Clone or download this repo
2. **For AI help:** Add your OpenAI API key in `config.js`
3. Open `index.html` in a web browser
4. Add daily logs—or click **Load Sample Data** to see how it works
5. Click **Analyze Patterns** to view insights
6. Click the **Help** button (bottom-right) for AI assistance

**Deploy on Vercel:**
1. Push to GitHub and import the repo in Vercel
2. Add **OPENAI_API_KEY** in Project Settings → Environment Variables
3. Deploy — the AI chatbot will use the serverless API and your env var

---

## Tech Stack

- **HTML / CSS / JavaScript** – Vanilla front-end
- **Chart.js** – Visualizations
- **OpenAI API** – AI help chatbot (gpt-4o-mini)
- **localStorage** – Client-side data persistence

---

## Important Note

This tool does **not** diagnose autism or predict behavior with certainty. It is designed to help caregivers notice patterns that may support better routine planning and understanding. Always consult healthcare providers for medical or behavioral advice.

---

## License

MIT
