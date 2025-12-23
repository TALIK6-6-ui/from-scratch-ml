# from-scratch-ml
Implementing Naive Bayes (Multinomial &amp; Bernoulli) from scratch in NumPy. Includes data preprocessing, model evaluation, and comparison with sklearn.
# Relearning Machine Learning — From the Ground Up

> “I used `sklearn.naive_bayes` for months... until I realized I couldn’t explain why Laplace smoothing mattered on a whiteboard.  
> So I decided to build it myself.”

This repo is my notebook as I reimplement core ML algorithms **without black-box libraries** — not to be better than sklearn,  
but to **finally understand** what’s happening inside.

I’m starting with **Naive Bayes**, using two datasets I find fascinating:
- 🍄 The Mushroom dataset: *Can you really tell a poisonous mushroom just by its gill color?*
- 📰 The AG News dataset: *How do words become signals for categories?*

---

## Why I’m Doing This

As someone who’s worked with ML in academic and prototyping settings, I’ve often leaned on high-level APIs.  
But real trust in a model comes from **knowing its assumptions, limits, and failure modes**.

By coding:
- Multinomial Naive Bayes (for word counts)
- Bernoulli Naive Bayes (for binary feature presence)

...from scratch in NumPy, I’m confronting questions like:
- What happens when a word never appears in training but shows up in test?
- Why does Bernoulli NB struggle with long documents?
- How does class imbalance silently skew probability estimates?

This isn’t production code (yet).  
It’s **thinking made visible**.

---

## What’s Inside

- `notebooks/`  
  My explorations — messy at first, refined over time. Includes side-by-side comparisons with sklearn,  
  but the focus is on *interpretation*, not benchmarking.

- `src/naive_bayes.py`  
  Clean, commented implementations. Every line exists to answer a question I had.

- `data/`  
  Raw and preprocessed versions — with notes on why I chose certain encodings or splits.

---

## A Note on “From Scratch”

I’m **not avoiding sklearn out of pride**. In fact, I use it daily.  
But I believe you should only use a tool confidently when you understand what it *hides*.

This is my way of lifting the hood.

---

## If You’re Reading This…

…maybe you’ve felt the same gap between using ML and *understanding* it.  
I’d love to hear what you’re rebuilding.  

— Touseef
